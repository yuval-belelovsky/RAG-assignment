import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from data_pipeline import process_small_dataset

# Load environment variables from the .env file safely
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Get configuration keys and clean hidden spaces/newlines using .strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY").strip() if os.getenv("PINECONE_API_KEY") else None
PINECONE_HOST = os.getenv("PINECONE_HOST").strip() if os.getenv("PINECONE_HOST") else None
AI_API_KEY = os.getenv("AI_API_KEY").strip() if os.getenv("AI_API_KEY") else None

# Debug print to verify the API key is loaded properly
if PINECONE_API_KEY:
    print(f"🔑 Pinecone API Key loaded successfully! Starts with: {PINECONE_API_KEY[:4]}...")
else:
    print("❌ ERROR: Could not find or read .env file!")

# Connect to the official Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


def get_embedding(text):
    """
    Sends text to the course API and returns a 1536-dimensional vector.
    """
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
        print(f"Error fetching embedding: {e}")
        return None


def upload_chunks_to_pinecone(chunks):
    """
    Converts text chunks into embeddings and uploads them to Pinecone in batches.
    """
    print(f"Starting to upload {len(chunks)} chunks to Pinecone...")

    vectors_to_upsert = []

    for i, chunk in enumerate(chunks):
        # Generate the embedding vector for the current chunk text
        vector = get_embedding(chunk['chunk'])

        if vector is None:
            print(f"Skipping chunk {chunk['id']} due to embedding error.")
            continue

        # Structure the data for Pinecone: (ID, Vector, Metadata)
        vectors_to_upsert.append((
            chunk['id'],
            vector,
            {
                "article_id": chunk['article_id'],
                "title": chunk['title'],
                "url": chunk['url'],
                "chunk": chunk['chunk']  # Storing the raw text to retrieve it later during RAG queries
            }
        ))

        # Upload to Pinecone in batches of 20 to avoid network overload
        if len(vectors_to_upsert) >= 20 or i == len(chunks) - 1:
            try:
                index.upsert(vectors=vectors_to_upsert)
                print(f"Successfully uploaded batch up to index {i}")
                vectors_to_upsert = []  # Reset the batch list
            except Exception as e:
                print(f"Error uploading batch to Pinecone: {e}")
                return False

    print("Finished uploading all test chunks successfully!")
    return True


if __name__ == "__main__":
    # 1. Load and process only the first 5 articles for safe testing
    test_chunks = process_small_dataset("medium-english-50mb.csv", limit=5)

    # 2. Run the pipeline to embed and upload data
    if test_chunks:
        upload_chunks_to_pinecone(test_chunks)