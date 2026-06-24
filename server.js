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
    const pipelineStart = Date.now();
    const metrics = { stages: {} };

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

    // ── Stage 1: Multi-Query ──────────────────────────────────────────────────
    let t = Date.now();
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

    metrics.stages.multiQuery = {
      ms: Date.now() - t,
      queries: altQueries,
    };

    // ── Stage 2: Retrieval ────────────────────────────────────────────────────
    t = Date.now();
    const baseRetriever = vectorStore.asRetriever({ k: 5 });
    const allQueries = [question, ...altQueries];
    const retrieved = await Promise.all(allQueries.map((q) => baseRetriever.invoke(q)));

    const seen = new Set();
    const candidates = retrieved.flat().filter((doc) => {
      if (seen.has(doc.pageContent)) return false;
      seen.add(doc.pageContent);
      return true;
    });

    metrics.stages.retrieval = {
      ms: Date.now() - t,
      queriesRun: allQueries.length,
      candidatesFound: candidates.length,
    };

    // ── Stage 3: Reranking ────────────────────────────────────────────────────
    t = Date.now();
    let reranked = candidates;
    let rerankScores = [];
    try {
      const rerankResponse = await client.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0,
        messages: [
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
        ],
        response_format: { type: "json_object" },
      });
      const parsed = JSON.parse(rerankResponse.choices[0].message.content);
      const scores = Array.isArray(parsed) ? parsed : Object.values(parsed)[0];
      rerankScores = candidates.map((_, i) =>
        typeof scores[i] === "number" ? Math.round(scores[i] * 100) / 100 : 0
      );
      const scored = candidates.map((doc, i) => ({ doc, score: rerankScores[i] }));
      const kept = scored.filter((item) => item.score >= 0.3).sort((a, b) => b.score - a.score);
      reranked = kept.length > 0 ? kept.map((item) => item.doc) : candidates.slice(0, 5);
    } catch (_) {
      // fallback: unranked
    }

    metrics.stages.reranking = {
      ms: Date.now() - t,
      scores: rerankScores,
      keptCount: reranked.length,
      droppedCount: candidates.length - reranked.length,
    };

    // ── Stage 4: Context Compression ─────────────────────────────────────────
    t = Date.now();
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

    const finalChunks = compressedChunks.length > 0 ? compressedChunks : reranked;

    const totalInputChars = reranked.slice(0, 6).reduce((s, d) => s + d.pageContent.length, 0);
    const totalOutputChars = finalChunks.reduce((s, d) => s + d.pageContent.length, 0);
    metrics.stages.compression = {
      ms: Date.now() - t,
      inputChunks: reranked.slice(0, 6).length,
      outputChunks: finalChunks.length,
      compressionRatio: totalInputChars > 0
        ? Math.round((1 - totalOutputChars / totalInputChars) * 100)
        : 0,
    };

    const context = finalChunks
      .map((c, i) => `[Chunk ${i + 1}${c.metadata?.loc?.pageNumber ? `, Page ${c.metadata.loc.pageNumber}` : ""}]\n${c.pageContent}`)
      .join("\n\n---\n\n");

    // ── Stage 5: Generation ───────────────────────────────────────────────────
    t = Date.now();
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

    metrics.stages.generation = {
      ms: Date.now() - t,
      contextChunks: finalChunks.length,
      promptChars: context.length,
    };
    metrics.totalMs = Date.now() - pipelineStart;

    res.json({ answer: response.choices[0].message.content, metrics });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
