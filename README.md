# Medium Article RAG Assistant

An efficient Retrieval-Augmented Generation (RAG) system built to answer questions about a corpus of roughly 7,600 Medium articles. The system relies strictly on the provided dataset context to eliminate hallucinations and deliver evidence-based responses.

## Live Demo & API Endpoints
* **Live URL:** `https://rag-assignment-blush.vercel.app/`
* **API Documentation (Swagger):** `https://rag-assignment-blush.vercel.app/docs`

## ⚙️ Hyperparameters
* **Chunk Size:** `800` (Max 1024 tokens)
* **Overlap Ratio:** `0.2` (Max 0.3)
* **Top-K:** `10` (Max 30)

## 🗄️ API Specifications
1. `POST /api/prompt` - Queries the system with a natural language question.
2. `GET /api/stats` - Returns the active configuration and hyperparameter settings.

## 🔒 Safety Fallback
If a user query cannot be answered using the dataset context, the system strictly responds with:
"I don't know based on the provided Medium articles data."
