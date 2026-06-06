import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables safely
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Configuration and clean keys using .strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY").strip() if os.getenv("PINECONE_API_KEY") else None
PINECONE_HOST = os.getenv("PINECONE_HOST").strip() if os.getenv("PINECONE_HOST") else None
AI_API_KEY = os.getenv("AI_API_KEY").strip() if os.getenv("AI_API_KEY") else None

# Initialize Pinecone clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

app = FastAPI()

# RAG Hyperparameters chosen within assignment constraints
CHUNK_SIZE = 800
OVERLAP_RATIO = 0.2
TOP_K = 10


# Define the precise incoming JSON structure
class PromptRequest(BaseModel):
    question: str


def get_query_embedding(text):
    """Generates a 1536-dimensional vector for the user's question."""
    url = "https://api.llmod.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "4UHRUIN-text-embedding-3-small",
        "input": text
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


def get_chat_response(system_prompt, user_prompt):
    """Calls the gpt-5-mini model with the augmented prompt contexts."""
    url = "https://api.llmod.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "4UHRUIN-gpt-5-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Chat generation failed: {str(e)}")


@app.get("/api/stats")
def get_stats():
    """Returns system hyperparameters matching exact required JSON field names."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }


@app.post("/api/prompt")
def query_rag(request: PromptRequest):
    """Processes RAG workflow and returns answers strictly based on context."""
    # 1. Convert user question into an embedding vector
    query_vector = get_query_embedding(request.question)

    # 2. Query Pinecone to retrieve the top closest vector contexts
    try:
        pinecone_res = index.query(
            vector=query_vector,
            top_k=TOP_K,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")

    # 3. Format context blocks for the JSON response and building the LLM context
    context_list = []
    context_passages = []

    for match in pinecone_res.get('matches', []):
        metadata = match.get('metadata', {})
        context_list.append({
            "article_id": metadata.get("article_id", ""),
            "title": metadata.get("title", ""),
            "chunk": metadata.get("chunk", ""),
            "score": match.get("score", 0.0)
        })
        # Format text block for the LLM prompt context
        context_passages.append(
            f"Title: {metadata.get('title')}\n"
            f"URL: {metadata.get('url')}\n"
            f"Passage: {metadata.get('chunk')}\n"
            f"---"
        )

    # 4. Construct the required strict system prompt from the instructions
    system_prompt = (
        "You are a Medium-article assistant that answers questions strictly and only "
        "based on the Medium articles dataset context provided to you (metadata and article passages). "
        "You must not use any external knowledge, the open internet, or information that is not explicitly "
        "contained in the retrieved context. If the answer cannot be determined from the provided context, "
        "respond: \"I don't know based on the provided Medium articles data.\"\n"
        "Always explain your answer using the given context, quoting or paraphrasing the relevant "
        "article passage or metadata when helpful."
    )

    # 5. Build the user prompt injection containing the retrieved facts
    context_str = "\n".join(context_passages)
    user_prompt = (
        f"Retrieved Context Passages:\n{context_str}\n\n"
        f"User Question: {request.question}"
    )

    # 6. Generate final answered response via gpt-5-mini
    llm_response = get_chat_response(system_prompt, user_prompt)

    # 7. Return exact JSON structural specifications outlined in the assignment
    return {
        "response": llm_response,
        "context": context_list,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }