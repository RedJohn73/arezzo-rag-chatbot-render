import os
import json
import time
import numpy as np
import faiss
from openai import OpenAI

from src.crawler import crawl_comune_arezzo
from src.chunker import chunk_text

# ============================================================
# CONFIG
# ============================================================

INDEX_DIR = "vectorstore"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "meta.json")
PROGRESS_FILE = "data/last_crawl.txt"

EMBEDDING_MODEL = "text-embedding-3-large"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Stato interno (progress bar)
progress = {"percent": 0, "step": "idle", "ready": False}


# ============================================================
# PROGRESS HANDLING
# ============================================================

def update_progress(p, step):
    progress["percent"] = p
    progress["step"] = step

def get_progress():
    return progress


# ============================================================
# INDEX STATE
# ============================================================

def index_exists():
    """Restituisce True se l'indice FAISS + meta.json esistono."""
    return os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)


# ============================================================
# INDEXING PIPELINE
# ============================================================

def build_index(pages):
    """Costruisce FAISS da una lista di pagine chunkate."""

    update_progress(5, "embedding")

    texts = [p["text"] for p in pages]

    vectors = []
    for i, t in enumerate(texts):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=t)
            vectors.append(resp.data[0].embedding)
        except Exception as e:
            print("[ERR] Embedding error:", e)
            continue

        # progress %
        update_progress(int(5 + (i / len(texts)) * 80), "embedding")

    vectors = np.array(vectors).astype("float32")

    # FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w") as f:
        json.dump(pages, f)

    update_progress(100, "done")
    progress["ready"] = True

    print("✔ FAISS index saved.")


def ensure_index_built_async():
    """Eseguito in background all’avvio dell’app."""
    if index_exists():
        progress["ready"] = True
        update_progress(100, "ready-existing")
        print("✔ Indice già esistente.")
        return

    print("⚠ Nessun indice → avvio crawling...")
    update_progress(0, "crawling")

    raw_pages = crawl_comune_arezzo()
    print(f"✔ Crawling completato: {len(raw_pages)} pagine Drupal.")

    update_progress(10, "chunking")
    pages_chunked = []

    for p in raw_pages:
        # Drupal spesso non ha "text": usa fallback
        text = p.get("text") or p.get("body") or ""
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for c in chunks:
            pages_chunked.append({"url": p["url"], "text": c})

    print(f"✔ Chunk generati: {len(pages_chunked)}")

    build_index(pages_chunked)


# ============================================================
# QUERY (ANSWER) PIPELINE
# ============================================================

def answer_question(query):
    """Ritorna la risposta sfruttando FAISS + OpenAI."""

    if not index_exists():
        return "⏳ L'indice è ancora in preparazione. Riprova tra poco."

    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "r") as f:
        meta = json.load(f)

    # Query embedding
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    q_vec = np.array(resp.data[0].embedding).astype("float32")

    # Ann search
    D, I = index.search(np.array([q_vec]), 5)

    retrieved = "\n\n".join(meta[i]["text"] for i in I[0] if i < len(meta))

    system_prompt = """
Sei ARIA, assistente istituzionale del Comune di Arezzo.
Rispondi in modo formale, chiaro e corretto, usando SOLO informazioni affidabili.
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": retrieved},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )

    return res.choices[0].message.content


# ============================================================
# MAIN (solo CLI)
# ============================================================

if __name__ == "__main__":
    ensure_index_built_async()
