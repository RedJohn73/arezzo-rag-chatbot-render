import os
import json
import time
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
PROGRESS_FILE = os.path.join(INDEX_DIR, "progress.json")

EMBEDDING_MODEL = "text-embedding-3-large"

client = OpenAI()


# ============================================================
# UTILS
# ============================================================

def save_progress(step: str, percent: int, ready: bool):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"step": step, "percent": percent, "ready": ready}, f)


def get_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {"step": "idle", "percent": 0, "ready": False}
    return json.load(open(PROGRESS_FILE))


def index_exists():
    return os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_texts(chunks: list):
    vectors = []

    for i, t in enumerate(chunks):
        try:
            save_progress("embedding", int((i / len(chunks)) * 100), False)

            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=t
            )
            vectors.append(resp.data[0].embedding)

        except Exception as e:
            print("[ERR] embedding failed:", e)
            continue

    return np.array(vectors).astype("float32")


# ============================================================
# COSTRUZIONE INDICE
# ============================================================

def build_index_async():
    """Funzione eseguita in thread separato."""
    try:
        save_progress("crawling", 0, False)
        pages = crawl_comune_arezzo()

        print("[INFO] Crawling:", len(pages), "pagine")

        # estrai testo da Drupal correttamente
        save_progress("chunking", 10, False)
        chunked = []
        for p in pages:
            text = p.get("text", "") or p.get("body", "")
            if not text.strip():
                continue
            for c in chunk_text(text):
                chunked.append({"url": p["url"], "text": c})

        print("[INFO] Chunks:", len(chunked))

        # vettori
        save_progress("embedding", 20, False)
        embeddings = embed_texts([c["text"] for c in chunked])

        # indice FAISS
        save_progress("faiss", 90, False)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, INDEX_FILE)

        with open(META_FILE, "w") as f:
            json.dump(chunked, f)

        save_progress("done", 100, True)
        print("[OK] Index built.")

    except Exception as e:
        print("[FATAL] build_index_async:", e)
        save_progress("error", 0, False)


# ============================================================
# END
# ============================================================
