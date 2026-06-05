import pandas as pd


def chunk_text(text, chunk_size=500, overlap=100):
    """
    פונקציה פשוטה לחיתוך טקסט לחתיכות עם חפיפה.
    בשלב ראשון נעבוד עם חיתוך לפי תווים/מילים בצורה פשוטה.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


def process_small_dataset(file_path, limit=5):
    """
    טעינת מספר מצומצם של מאמרים לצורך בדיקת הצינור (Pipeline) ללא עלות גבוהה.
    """
    print(f"Loading first {limit} articles from {file_path}...")
    # טעינת ה-CSV (אנחנו צריכים את העמודות: title, text, url, authors, timestamp, tags)
    df = pd.read_csv(file_path).head(limit)

    all_chunks = []

    # מעבר על המאמרים וחיתוך שלהם
    for index, row in df.iterrows():
        article_id = str(index)
        title = str(row['title'])
        text = str(row['text'])
        url = str(row['url'])

        # חיתוך הטקסט של המאמר לחתיכות
        text_chunks = chunk_text(text, chunk_size=500, overlap=100)

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
    # הרצה לבדיקה מקומית לראות שהחיתוך עובד
    chunks = process_small_dataset("medium-english-50mb.csv", limit=5)
    if chunks:
        print("Example chunk:", chunks[0])