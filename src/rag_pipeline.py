
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

# percorso persistente su Render
PERSISTENT_DIR = Path("/opt/render/project/data")
PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PERSISTENT_DIR

PAGES_JSON = DATA_DIR / "pages.json"
EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"
META_JSON = DATA_DIR / "meta.json"
LAST_CRAWL = DATA_DIR / "last_crawl.txt"

CRAWL_INTERVAL_HOURS = 24

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
# GESTIONE PROGRESSO
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
# LOGICA DI RICOSTRUZIONE INDICE
# ============================================================

def _should_rebuild() -> bool:
    if not LAST_CRAWL.exists():
        return True
    try:
        ts = float(LAST_CRAWL.read_text())
    except Exception:
        return True
    age_hours = (time.time() - ts) / 3600
    return age_hours >= CRAWL_INTERVAL_HOURS


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
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
    # normalizzazione
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    arr = arr / norms
    return arr


def _build_index():
    global index_embeddings, index_meta

    print("[INFO] Avvio crawling...")
    _set_progress("crawling", 0, 1)

    pages = crawl_comune_arezzo(
        max_pages=1200,
        max_depth=8,
        delay_seconds=0.15,
    )

    _set_progress("crawling", 1, 1)

    with open(PAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Crawling completato: {len(pages)} pagine.")

    chunks: list[str] = []
    meta: list[dict[str, Any]] = []

    total_pages = len(pages)
    _set_progress("chunking", 0, total_pages)

    for i, page in enumerate(pages):
        content_chunks = _chunk_text(page.get("content", ""))
        for ch in content_chunks:
            chunks.append(ch)
            meta.append(
                {
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "snippet": ch[:250],
                }
            )
        _set_progress("chunking", i + 1, total_pages)

    print(f"[INFO] Totale chunk generati: {len(chunks)}")

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
    """Da chiamare in un thread separato all'avvio dell'app.

    - Se esiste già un indice recente, lo carica.
    - Altrimenti esegue un crawl + rebuild completo.
    """
    global index_embeddings, index_meta

    try:
        if (
            EMBEDDINGS_NPY.exists()
            and META_JSON.exists()
            and not _should_rebuild()
        ):
            print("[INFO] Carico indice esistente da disco...")
            index_embeddings = np.load(EMBEDDINGS_NPY)
            with open(META_JSON, "r", encoding="utf-8") as f:
                index_meta = json.load(f)
            _progress["ready"] = True
            return

        print("[INFO] Ricostruzione completa dell'indice...")
        _build_index()

    except Exception as e:
        print("[ERR] Errore durante la costruzione dell'indice:", e)
        _progress["ready"] = False


def answer_question(question: str) -> str:
    """Risponde a una domanda usando RAG sul sito del Comune di Arezzo."""
    if not question or not question.strip():
        return "Per favore inserisci una domanda valida."

    if not _progress["ready"] or index_embeddings is None or index_meta is None:
        prog = get_progress()
        return (
            f"⏳ Sto ancora preparando l'indice ("
            f"{prog['percent']}% – step: {prog['step']}). "
            "Riprova tra qualche istante."
        )

    q = question.strip()

    # embedding della query
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    q_vec = resp.data[0].embedding
    q_vec = np.array(q_vec, dtype="float32")
    q_vec /= (np.linalg.norm(q_vec) + 1e-8)

    sims = np.dot(index_embeddings, q_vec)
    top_k = 6
    top_idx = np.argsort(-sims)[:top_k]

    context_blocks = []
    urls = []

    for idx in top_idx:
        idx_int = int(idx)
        entry = index_meta[idx_int]
        urls.append(entry.get("url", ""))
        context_blocks.append(
            f"TITOLO: {entry.get('title', '')}\n"
            f"URL: {entry.get('url', '')}\n"
            f"{entry.get('snippet', '')}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "Sei l'assistente istituzionale del Comune di Arezzo. "
                "Rispondi in italiano, in modo chiaro e conciso, usando SOLO "
                "le informazioni provenienti dai documenti forniti."
            ),
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
