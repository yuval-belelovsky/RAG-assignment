import pandas as pd


def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into chunks with a sliding window overlap."""
    chunks = []
    start = 0
    # Ensure stride is at least 1 to avoid an infinite loop
    stride = max(1, chunk_size - overlap)

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += stride
    return chunks


def process_small_dataset(file_path, limit=5, chunk_size=500, overlap_ratio=0.2):
    """Loads a subset of articles and dynamically cuts them based on settings."""
    print(f"Loading first {limit} articles from {file_path}...")
    print(f"Processing chunks with size: {chunk_size}, overlap ratio: {overlap_ratio}")

    # Load the CSV matching the strict requested schema
    # If limit is provided, take only the head, otherwise load the entire file
    if limit:
        df = pd.read_csv(file_path).head(limit)
    else:
        df = pd.read_csv(file_path)
    all_chunks = []

    # Calculate actual character/token overlap size from the ratio
    actual_overlap = int(chunk_size * overlap_ratio)

    for index, row in df.iterrows():
        article_id = str(index)
        title = str(row['title'])
        text = str(row['text'])
        url = str(row['url'])

        # Cut text using the dynamic parameters passed into the function
        text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=actual_overlap)

        for chunk_index, chunk_content in enumerate(text_chunks):
            all_chunks.append({
                "id": f"art_{article_id}_chk_{chunk_index}",
                "article_id": article_id,
                "title": title,
                "url": url,
                "chunk": chunk_content
            })

    print(f"Created {len(all_chunks)} chunks from {limit} articles.")
    return all_chunks


if __name__ == "__main__":
    chunks = process_small_dataset("medium-english-50mb.csv", limit=5)
    if chunks:
        print("Example chunk:", chunks[0])