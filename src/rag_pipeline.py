import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from .crawler import crawl_comune_arezzo

# ============================================================
# CONFIGURAZIONE DI BASE
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)   # <-- CREA AUTOMATICAMENTE /data

PAGES_JSON = DATA_DIR / "pages.json"
EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"
META_JSON = DATA_DIR / "meta.json"
LAST_CRAWL = DATA_DIR / "last_crawl.txt"

# Ricostruzione indice ogni 12 ore
CRAWL_INTERVAL_HOURS = 12

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# VARIABILI GLOBALI
# ============================================================

index_embeddings = None
index_meta: list[dict[str, Any]] | None = None

_progress = {
    "step": "idle",
    "current": 0,
    "total": 1,
    "ready": False,
}

# ============================================================
# PROGRESSO
# ============================================================

def get_progress() -> dict:
    total = max(_progress["total"], 1)
    percent = int((_progress["current"] / total) * 100)
    return {
        "percent": percent,
        "step": _progress["step"],
        "ready": _progress["ready"],
    }


def _set_progress(step: str, current: int, total: int):
    _progress["step"] = step
    _progress["current"] = current
    _progress["total"] = max(total, 1)


# ============================================================
# LOGICA DI RICOSTRUZIONE DELL'INDICE
# ============================================================

def _should_rebuild() -> bool:
    """Ritorna True se l’ultimo crawl è più vecchio di X ore."""
    if not LAST_CRAWL.exists():
        return True

    try:
        ts = float(LAST_CRAWL.read_text())
    except Exception:
        return True

    age_hours = (time.time() - ts) / 3600
    print(f"[INFO] Età indice: {age_hours:.2f} ore")

    return age_hours >= CRAWL_INTERVAL_HOURS


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        end = min(i + max_chars, n)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = end - overlap

    return chunks


def _embed_chunks(chunks: List[str]) -> np.ndarray:
    _set_progress("embedding", 0, len(chunks))

    if not chunks:
        return np.zeros((0, 512), dtype="float32")

    vectors = []
    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
        except Exception as e:
            print("[ERR] Embedding error:", e)
            continue

        for item in resp.data:
            vectors.append(item.embedding)

        _set_progress("embedding", i + len(batch), len(chunks))

    arr = np.array(vectors, dtype="float32")
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    return arr


def _build_index():
    """Esegue crawling + chunk + embedding + salvataggio."""
    global index_embeddings, index_meta

    print("[INFO] Avvio crawling...")
    _set_progress("crawling", 0, 1)

    pages = crawl_comune_arezzo(
        max_pages=1200,
        max_depth=8,
        delay_seconds=0.10,
    )

    _set_progress("crawling", 1, 1)

    with open(PAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Crawling completato: {len(pages)} pagine.")

    chunks = []
    meta = []

    total_pages = len(pages)
    _set_progress("chunking", 0, total_pages)

    for i, page in enumerate(pages):
        text = page.get("content", "")
        for ch in _chunk_text(text):
            chunks.append(ch)
            meta.append({
                "url": page.get("url", ""),
                "title": page.get("title", ""),
                "snippet": ch[:250],
            })

        _set_progress("chunking", i + 1, total_pages)

    print(f"[INFO] Chunk generati: {len(chunks)}")

    embeddings = _embed_chunks(chunks)

    np.save(EMBEDDINGS_NPY, embeddings)
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LAST_CRAWL.write_text(str(time.time()))

    index_embeddings = embeddings
    index_meta = meta

    _progress["ready"] = True
    print("[INFO] Indicizzazione completata.")


def ensure_index_built_async():
    """Usato da app.py — lancia in background il rebuild se necessario."""
    global index_embeddings, index_meta

    try:
        if EMBEDDINGS_NPY.exists() and META_JSON.exists() and not _should_rebuild():
            print("[INFO] Carico indice esistente...")
            index_embeddings = np.load(EMBEDDINGS_NPY)
            with open(META_JSON, "r", encoding="utf-8") as f:
                index_meta = json.load(f)
            _progress["ready"] = True
            return

        print("[INFO] Ricostruzione indice necessaria.")
        _build_index()

    except Exception as e:
        print("[ERR] Errore ricostruzione indice:", e)
        _progress["ready"] = False


# ============================================================
# RAG QUERY
# ============================================================

def answer_question(question: str) -> str:
    if not question or not question.strip():
        return "Inserisci una domanda valida."

    if not _progress["ready"]:
        p = get_progress()
        return f"⏳ Indice non pronto ({p['percent']}% – step: {p['step']}). Riprova più tardi."

    q = question.strip()

    # embedding query
    emb = client.embeddings.create(model=EMBED_MODEL, input=[q])
    q_vec = np.array(emb.data[0].embedding, dtype="float32")
    q_vec /= (np.linalg.norm(q_vec) + 1e-8)

    sims = np.dot(index_embeddings, q_vec)
    top_k = 6
    idxs = np.argsort(-sims)[:top_k]

    context_blocks = []
    urls = []

    for idx in idxs:
        entry = index_meta[int(idx)]
        urls.append(entry["url"])
        context_blocks.append(
            f"TITOLO: {entry['title']}\nURL: {entry['url']}\n{entry['snippet']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content":
                "Sei l'assistente istituzionale del Comune di Arezzo. "
                "Usa SOLO le informazioni nel contesto fornito."
        },
        {"role": "system", "content": context},
        {"role": "user", "content": q},
    ]

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    answer = completion.choices[0].message.content
    answer += "\n\nFonti:\n" + "\n".join(f"- {u}" for u in urls if u)

    return answer


# ============================================================
# ESECUZIONE LOCALE
# ============================================================

if __name__ == "__main__":
    print("=== ARIA RAG PIPELINE ===")
    print("Controllo indice...")

    ensure_index_built_async()

    print("Pronto.")
