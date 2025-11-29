def chunk_text(text, max_len=800):
    words = text.split()
    chunks = []
    current = []

    for w in words:
        current.append(w)
        if len(current) >= max_len:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks
