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
    const chunks = retrieved.flat().filter((doc) => {
      if (seen.has(doc.pageContent)) return false;
      seen.add(doc.pageContent);
      return true;
    });

    const context = chunks
      .map((c, i) => `[Chunk ${i + 1}${c.metadata?.loc?.pageNumber ? `, Page ${c.metadata.loc.pageNumber}` : ""}]\n${c.pageContent}`)
      .join("\n\n---\n\n");

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are an AI assistant that answers questions strictly based on the document context below. Do not use outside knowledge. If the answer is not in the context, say "I couldn't find that in the document."\n\nDocument context:\n${context}`,
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
