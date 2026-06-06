# Medium Article RAG Assistant

[cite_start]An efficient Retrieval-Augmented Generation (RAG) system built to answer questions about a corpus of roughly 7,600 Medium articles[cite: 4, 8]. [cite_start]The system relies strictly on the provided dataset context to eliminate hallucinations and deliver evidence-based responses[cite: 5].

## 🚀 Live Demo & API Endpoints
* [cite_start]**Live URL:** `https://your-project-name.vercel.app` [cite: 61]
* **API Documentation (Swagger):** `https://your-project-name.vercel.app/docs`

## ⚙️ Hyperparameters
* [cite_start]**Chunk Size:** `800` (Max 1024 tokens) [cite: 42]
* [cite_start]**Overlap Ratio:** `0.2` (Max 0.3) [cite: 43]
* [cite_start]**Top-K:** `10` (Max 30) [cite: 44]

## 🗄️ API Specifications
1. [cite_start]`POST /api/prompt` - Queries the system with a natural language question[cite: 64, 65].
2. [cite_start]`GET /api/stats` - Returns the active configuration and hyperparameter settings[cite: 89, 90].

## 🔒 Safety Fallback
[cite_start]If a user query cannot be answered using the dataset context, the system strictly responds with[cite: 49, 51]:
> [cite_start]`"I don't know based on the provided Medium articles data."` [cite: 51]
