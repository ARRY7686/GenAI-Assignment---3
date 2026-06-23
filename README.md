# NotebookLM Clone — Advanced RAG-Powered Document Chat

Upload any PDF or plain text file and have a conversation with it. Built on a full Advanced RAG pipeline: chunking → embedding → vector storage → multi-query retrieval → grounded generation.

## Stack

| Layer | Technology |
|---|---|
| Server | Node.js + Express |
| Document loading | LangChain PDFLoader / TextLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector DB | Qdrant |
| Retrieval | LangChain `MultiQueryRetriever` |
| Generation | OpenAI `gpt-4o-mini` |
| UI | Vanilla HTML/CSS/JS |

## RAG Pipeline

1. **Ingest** — user uploads a PDF or `.txt` file via the web UI
2. **Chunk** — `RecursiveCharacterTextSplitter` splits the document into 1000-character chunks with 200-character overlap, preserving sentence boundaries
3. **Embed** — each chunk is converted to a vector using `text-embedding-3-large`
4. **Store** — vectors are stored in a Qdrant collection named by a unique session UUID
5. **Retrieve (Advanced)** — at query time, `MultiQueryRetriever` uses an LLM to generate multiple semantic variations of the user's question, retrieves documents for all queries, and takes the unique union. This dramatically improves retrieval accuracy by overcoming wording variations.
6. **Generate** — the retrieved chunks are injected into the system prompt; `gpt-4o-mini` answers strictly from that context

---

## Local Setup

### Prerequisites
- Node.js 18+
- Docker (for local Qdrant) **or** a [Qdrant Cloud](https://cloud.qdrant.io) account
- GitHub Personal Access Token (for inference via GitHub Models)

### 1. Install dependencies
```bash
npm install
```

### 2. Start Qdrant locally (skip if using Qdrant Cloud)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your keys (GITHUB_TOKEN, etc.)
```

### 4. Run
```bash
npm start
# Open http://localhost:3000
```

---

## Deployment

### Qdrant Cloud (free tier)
1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster → copy the **URL** and **API key**
3. Set `QDRANT_URL` and `QDRANT_API_KEY` in your deployment environment

### Render.com (free tier)
1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect the repo
3. Set:
   - **Build Command:** `npm install`
   - **Start Command:** `npm start`
4. Add environment variables:
   - `GITHUB_TOKEN`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
5. Deploy — Render provides a public HTTPS URL

---

## Project Structure

```
assignment-3/
├── server.js          # Express server + full RAG pipeline
├── public/
│   └── index.html     # Single-page chat UI
├── package.json
├── .env.example
└── README.md
```
