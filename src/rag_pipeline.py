import os
import json
import time
import threading
import numpy as np
import faiss

from openai import OpenAI
from .crawler import crawl_comune_arezzo
from .chunker import chunk_text

# ============================================================
# CONFIG
# ============================================================

INDEX_DIR = "vectorstore"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "meta.json")
LAST_CRAWL = "data/last_crawl.txt"

EMBED_MODEL = "text-embedding-3-large"
client = OpenAI()

# Stato interno per progress
_progress = {
    "percent": 0,
    "step": "idle",
    "ready": False
}


# ============================================================
# CHECK SE INDICE ESISTE
# ============================================================

def index_exists():
    return os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)


def get_progress():
    return _progress


# ============================================================
# FUNZIONE EMBEDDING
# ============================================================

def embed(text):
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(r.data[0].embedding, dtype="float32")


# ============================================================
# COSTRUZIONE INDICE
# ============================================================

def build_index():
    global _progress
    _progress.update({"percent": 0, "step": "crawl", "ready": False})

    pages_raw = crawl_comune_arezzo()
    print(f"[INFO] Crawling completato: {len(pages_raw)} pagine.")

    all_chunks = []
    for p in pages_raw:
        if "html" in p:
            text = p["html"]
        elif "text" in p:
            text = p["text"]
        else:
            continue

        for c in chunk_text(text):
            all_chunks.append({"url": p["url"], "text": c})

    print(f"[INFO] Chunk generati: {len(all_chunks)}")

    _progress.update({"percent": 30, "step": "embeddings"})

    vectors = []

    for idx, chunk in enumerate(all_chunks):
        try:
            vec = embed(chunk["text"])
            vectors.append(vec)
        except Exception as e:
            print("[ERR] embedding:", e)
            continue

        if idx % 50 == 0:
            _progress["percent"] = 30 + int((idx / len(all_chunks)) * 60)

    vectors = np.array(vectors, dtype="float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    json.dump(all_chunks, open(META_FILE, "w"))

    _progress.update({"percent": 100, "step": "done", "ready": True})
    print("[OK] Indice FAISS salvato.")


# ============================================================
# COSTRUZIONE ASINCRONA (PER Render)
# ============================================================

def build_index_async():
    if index_exists():
        _progress.update({"percent": 100, "step": "done", "ready": True})
        return

    print("[INFO] Avvio indicizzazione async...")
    threading.Thread(target=build_index, daemon=True).start()
