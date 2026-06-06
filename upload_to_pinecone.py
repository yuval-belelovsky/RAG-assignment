import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from data_pipeline import process_small_dataset
import time

# Load environment variables from the .env file safely
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Get configuration keys and clean hidden spaces/newlines using .strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY").strip() if os.getenv("PINECONE_API_KEY") else None
PINECONE_HOST = os.getenv("PINECONE_HOST").strip() if os.getenv("PINECONE_HOST") else None
AI_API_KEY = os.getenv("AI_API_KEY").strip() if os.getenv("AI_API_KEY") else None

# Debug print to verify the API key is loaded properly
if PINECONE_API_KEY:
    print(f"Pinecone API Key loaded successfully! Starts with: {PINECONE_API_KEY[:4]}...")
else:
    print("ERROR: Could not find or read .env file!")

# Connect to the official Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


def get_embedding(texts_list, retries=3, delay=2):
    """
    Sends a list of texts to the course API with an automatic retry mechanism for 502 errors.
    """
    url = "https://api.llmod.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "4UHRUIN-text-embedding-3-small",
        "input": texts_list
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            # If a server error occurs (like 502), trigger the exception to retry
            response.raise_for_status()

            data = response.json()
            return [item["embedding"] for item in data["data"]]

        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed for batch: {e}")
            if attempt < retries - 1:
                print(f"Waiting {delay} seconds before trying again...")
                time.sleep(delay)  # Wait for the server load to decrease
            else:
                print("All retry attempts failed for this batch.")
                return None

def upload_chunks_to_pinecone(chunks):
    """
    Converts text chunks into embeddings and uploads them to Pinecone in batches of 100.
    """
    print(f"Starting to upload {len(chunks)} chunks to Pinecone...")
    batch_size = 100
    total_chunks = len(chunks)

    # Iterate over the chunks list in steps of 100
    for i in range(0, total_chunks, batch_size):
        # Slice the current batch of 100 chunks
        batch_chunks = chunks[i:i + batch_size]

        # Extract only the raw text from each chunk in this batch
        texts_to_embed = [c['chunk'] for c in batch_chunks]

        # Fetch all embeddings for this batch in a single API call
        embeddings = get_embedding(texts_to_embed)

        if not embeddings or len(embeddings) != len(batch_chunks):
            print(f"Skipping batch starting at index {i} due to embedding error.")
            continue

        vectors_to_upsert = []

        # Construct the exact data structure expected by Pinecone
        for idx, chunk in enumerate(batch_chunks):
            vectors_to_upsert.append((
                chunk['id'],
                embeddings[idx],  # The matching vector from the API response
                {
                    "article_id": chunk['article_id'],
                    "title": chunk['title'],
                    "url": chunk['url'],
                    "chunk": chunk['chunk']
                }
            ))

        # Upload the entire batch of 100 vectors to Pinecone at once
        try:
            index.upsert(vectors=vectors_to_upsert)
            print(f"Successfully uploaded batch up to index {min(i + batch_size, total_chunks)}")
        except Exception as e:
            print(f"Error uploading batch to Pinecone at index {i}: {e}")
            return False

    print("Finished uploading all chunks successfully!")
    return True

if __name__ == "__main__":
    # Import the final winning parameters from your server config
    from main import CHUNK_SIZE, OVERLAP_RATIO

    print(f"FINAL RUN: Preparing FULL dataset with parameters: Chunk Size = {CHUNK_SIZE}, Overlap = {OVERLAP_RATIO}")


    # Load and process the dataset
    test_chunks = process_small_dataset(
        "medium-english-50mb.csv",
        limit=None,  # None means load all rows in the dataset!
        chunk_size=CHUNK_SIZE,
        overlap_ratio=OVERLAP_RATIO
    )

    # Embed and upload everything to the Pinecone cloud
    if test_chunks:
        upload_chunks_to_pinecone(test_chunks)