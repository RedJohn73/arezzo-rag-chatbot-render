def chunk_text(text, max_len=1200, overlap=150):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_len
        chunks.append(text[start:end])
        start = end - overlap

    return chunks
