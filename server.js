import "dotenv/config";
import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { v4 as uuidv4 } from "uuid";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import pdfParse from "pdf-parse/lib/pdf-parse.js";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAI } from "openai";

const GITHUB_BASE_URL = "https://models.inference.ai.azure.com";
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Ensure uploads dir exists (/tmp is the only writable dir in serverless envs)
const UPLOADS_DIR = process.env.UPLOADS_DIR || "/tmp/uploads";
fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const upload = multer({ dest: UPLOADS_DIR });

const qdrantConfig = {
  url: process.env.QDRANT_URL || "http://localhost:6333",
  ...(process.env.QDRANT_API_KEY && { apiKey: process.env.QDRANT_API_KEY }),
};

// POST /upload — ingest a PDF or .txt file into its own Qdrant collection
app.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded." });

  const filePath = req.file.path;
  const ext = path.extname(req.file.originalname).toLowerCase();

  if (ext !== ".pdf" && ext !== ".txt") {
    fs.unlinkSync(filePath);
    return res.status(400).json({ error: "Only PDF and .txt files are supported." });
  }

  const sessionId = uuidv4();

  try {
    // 1. Load
    let docs;
    if (ext === ".pdf") {
      const buffer = fs.readFileSync(filePath);
      const result = await pdfParse(buffer);
      docs = [{ pageContent: result.text, metadata: { source: req.file.originalname } }];
    } else {
      const text = fs.readFileSync(filePath, "utf-8");
      docs = [{ pageContent: text, metadata: { source: req.file.originalname } }];
    }

    // 2. Chunk — RecursiveCharacterTextSplitter with 1000-char chunks, 200-char overlap
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunks = await splitter.splitDocuments(docs);

    // 3. Embed + Store into Qdrant collection named by sessionId
    const embeddings = new OpenAIEmbeddings({
      model: "text-embedding-3-large",
      apiKey: GITHUB_TOKEN,
      configuration: { baseURL: GITHUB_BASE_URL },
    });
    await QdrantVectorStore.fromDocuments(chunks, embeddings, {
      ...qdrantConfig,
      collectionName: sessionId,
    });

    fs.unlinkSync(filePath);
    res.json({ sessionId, chunks: chunks.length });
  } catch (err) {
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// POST /chat — retrieve relevant chunks and generate a grounded answer
app.post("/chat", async (req, res) => {
  const { question, sessionId } = req.body;
  if (!question || !sessionId) {
    return res.status(400).json({ error: "question and sessionId are required." });
  }

  try {
    const embeddings = new OpenAIEmbeddings({
      model: "text-embedding-3-large",
      apiKey: GITHUB_TOKEN,
      configuration: { baseURL: GITHUB_BASE_URL },
    });
    const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
      ...qdrantConfig,
      collectionName: sessionId,
    });

    const client = new OpenAI({ baseURL: GITHUB_BASE_URL, apiKey: GITHUB_TOKEN });

    // Advanced RAG: Multi-Query Retrieval — generate 3 alternative phrasings of the
    // user question, retrieve chunks for each, then deduplicate before answering.
    const baseRetriever = vectorStore.asRetriever({ k: 5 });

    const mqResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.7,
      messages: [
        {
          role: "system",
          content:
            "Generate 3 alternative phrasings of the user's question to improve document retrieval. " +
            "Output exactly 3 lines, one question per line, no numbering or bullet points.",
        },
        { role: "user", content: question },
      ],
    });

    const altQueries = mqResponse.choices[0].message.content
      .split("\n")
      .map((q) => q.trim())
      .filter(Boolean)
      .slice(0, 3);

    const allQueries = [question, ...altQueries];
    const retrieved = await Promise.all(allQueries.map((q) => baseRetriever.invoke(q)));

    // Deduplicate by pageContent
    const seen = new Set();
    const candidates = retrieved.flat().filter((doc) => {
      if (seen.has(doc.pageContent)) return false;
      seen.add(doc.pageContent);
      return true;
    });

    // Advanced RAG: Reranking — score each candidate chunk for relevance to the
    // original question in a single batched LLM call, then keep the top chunks.
    const rerankMessages = [
      {
        role: "system",
        content:
          "You are a relevance scoring engine. Given a question and a list of text chunks, " +
          "output a JSON array of numbers (one per chunk, in order) representing relevance " +
          "scores from 0.0 (irrelevant) to 1.0 (highly relevant). Output ONLY the JSON array, no explanation.",
      },
      {
        role: "user",
        content:
          `Question: ${question}\n\nChunks:\n` +
          candidates.map((c, i) => `[${i}] ${c.pageContent}`).join("\n\n"),
      },
    ];

    let reranked = candidates; // fallback: keep original order
    try {
      const rerankResponse = await client.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0,
        messages: rerankMessages,
        response_format: { type: "json_object" },
      });
      // The model may wrap the array in an object; handle both cases
      const parsed = JSON.parse(rerankResponse.choices[0].message.content);
      const scores = Array.isArray(parsed) ? parsed : Object.values(parsed)[0];
      reranked = candidates
        .map((doc, i) => ({ doc, score: typeof scores[i] === "number" ? scores[i] : 0 }))
        .sort((a, b) => b.score - a.score)
        .filter((item) => item.score >= 0.3)   // drop clearly irrelevant chunks
        .map((item) => item.doc);
      if (reranked.length === 0) reranked = candidates.slice(0, 5); // safety: never empty
    } catch (_) {
      // If reranking fails, continue with unranked candidates
    }

    // Advanced RAG: Context Compression — for each reranked chunk, extract only the
    // sentences that are directly relevant to the question, reducing noise in the prompt.
    const compressionPromises = reranked.slice(0, 6).map((doc) =>
      client.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0,
        messages: [
          {
            role: "system",
            content:
              "Extract and return only the sentences from the text below that are directly " +
              "relevant to answering the question. Preserve the original wording exactly. " +
              "If nothing is relevant, return an empty string.",
          },
          {
            role: "user",
            content: `Question: ${question}\n\nText:\n${doc.pageContent}`,
          },
        ],
      })
    );

    const compressionResults = await Promise.all(compressionPromises);
    const compressedChunks = compressionResults
      .map((r, i) => ({
        pageContent: r.choices[0].message.content.trim(),
        metadata: reranked[i].metadata,
      }))
      .filter((c) => c.pageContent.length > 0);

    const chunks = compressedChunks.length > 0 ? compressedChunks : reranked;

    const context = chunks
      .map((c, i) => `[Chunk ${i + 1}${c.metadata?.loc?.pageNumber ? `, Page ${c.metadata.loc.pageNumber}` : ""}]\n${c.pageContent}`)
      .join("\n\n---\n\n");

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.3,
      messages: [
        {
          role: "system",
          content: `You are a helpful assistant that answers questions based on the provided document context. Use the context to synthesize a clear, accurate answer. You may infer and summarize from the context — you do not need a verbatim match. Only say "I couldn't find that in the document." if the context contains absolutely no relevant information for the question.\n\nDocument context:\n${context}`,
        },
        { role: "user", content: question },
      ],
    });

    res.json({ answer: response.choices[0].message.content });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
