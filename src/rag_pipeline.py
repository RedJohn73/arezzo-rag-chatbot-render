import os
import json
import numpy as np
import faiss
from openai import OpenAI
from .crawler import crawl_comune_arezzo
from .chunker import chunk_text

INDEX_DIR = "vectorstore"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "meta.json")

EMBEDDING_MODEL = "text-embedding-3-large"

client = OpenAI()  # prende OPENAI_API_KEY in automatico


def build_index(pages):
    texts = [p["text"] for p in pages]

    vectors = []
    for t in texts:
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=t
            )
            vectors.append(resp.data[0].embedding)
        except Exception as e:
            print("[ERR] Embedding error:", e)
            continue

    vectors = np.array(vectors).astype("float32")

    if len(vectors.shape) != 2:
        print("[ERR] Errore embedding → vettori vuoti o malformati")
        return

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w") as f:
        json.dump(pages, f)

    print("[OK] FAISS index saved.")


def load_index():
    if not os.path.exists(INDEX_FILE):
        return None
    return faiss.read_index(INDEX_FILE)


def index_exists():
    return os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)


def main():
    print("=== ARIA RAG PIPELINE ===")

    if index_exists():
        print("[INFO] Indice già esistente ✔")
        return

    print("[INFO] Nessun indice → avvio crawling...")
    pages_raw = crawl_comune_arezzo()

    print(f"[INFO] Crawling completato: {len(pages_raw)} pagine.")

    pages_chunked = []
    for p in pages_raw:
        chunks = chunk_text(p["text"])
        for c in chunks:
            pages_chunked.append({"url": p["url"], "text": c})

    print(f"[INFO] Chunk generati: {len(pages_chunked)}")

    print("[INFO] Avvio creazione indice FAISS...")
    build_index(pages_chunked)

    print("Pronto.")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
