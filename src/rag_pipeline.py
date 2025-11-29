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

EMBED_MODEL = "text-embedding-3-large"

client = OpenAI()


def build_index(chunks):
    print("[INFO] Embedding chunks:", len(chunks))

    vectors = []
    meta = []

    for c in chunks:
        try:
            r = client.embeddings.create(
                model=EMBED_MODEL,
                input=c["text"]
            )
            vectors.append(r.data[0].embedding)
            meta.append({"url": c["url"], "text": c["text"][:200]})
        except Exception as e:
            print("[ERR] embed:", e)
            continue

    vectors = np.array(vectors).astype("float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w") as f:
        json.dump(meta, f)

    print("[OK] FAISS index saved.")


def main():
    print("=== ARIA RAG PIPELINE ===")

    if os.path.exists(INDEX_FILE):
        print("[INFO] Indice già presente, fine.")
        return

    print("[INFO] Nessun indice → crawling Drupal...")
    pages = crawl_comune_arezzo()

    print(f"[INFO] Crawling completato → {len(pages)} pagine.")

    chunks = []
    for p in pages:
        parts = chunk_text(p["content"])
        for c in parts:
            chunks.append({"url": p["url"], "text": c})

    print(f"[INFO] Chunk generati: {len(chunks)}")

    build_index(chunks)
    print("Pronto.")


if __name__ == "__main__":
    main()
