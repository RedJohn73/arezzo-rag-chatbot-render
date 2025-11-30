import os

# 🔥 Fix critici: elimina completamente audio + analytics
os.environ["GRADIO_DISABLE_ANALYTICS"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["GRADIO_AUDIO_DEPENDENCIES"] = "0"

import gradio as gr
import threading
import time
import json

from src.rag_query import answer_question
from src.rag_pipeline import (
    index_exists,
    build_full_index_async,
)

# ============================================================
# TRACKING DELLO STATO DI INDICIZZAZIONE
# ============================================================

PROGRESS_FILE = "data/last_crawl.txt"

def get_progress():
    """
    Restituisce:
    {
        "percent": int,
        "step": "idle" | "crawling" | "embedding" | ...
        "ready": bool
    }
    """
    if index_exists():
        return {"percent": 100, "step": "done", "ready": True}

    if not os.path.exists(PROGRESS_FILE):
        return {"percent": 0, "step": "idle", "ready": False}

    try:
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
        return data
    except:
        return {"percent": 0, "step": "idle", "ready": False}


# ============================================================
# AVVIO INDICIZZAZIONE IN BACKGROUND
# ============================================================

index_started = False

def ensure_index_built_async():
    """
    Esegue il crawling + embeddings + FAISS in un thread separato,
    aggiornando PROGRESS_FILE mano a mano.
    """
    build_full_index_async()   # 🔥 questa è la funzione ufficiale della pipeline


def start_index_once():
    global index_started
    if not index_started:
        index_started = True
        threading.Thread(target=ensure_index_built_async, daemon=True).start()
    return progress_text()


def progress_text():
    prog = get_progress()
    if prog["ready"]:
        return "📗 Indice pronto ✔️"
    return f"🟧 Preparazione indice… {prog['percent']}%  (step: {prog['step']})"


# ============================================================
# FUNZIONE DI RISPOSTA
# ============================================================

def handle_question(q: str):
    if not q.strip():
        return "Inserisci una domanda."

    try:
        res = answer_question(q)
        return res
    except Exception as e:
        return f"❌ Errore interno: {e}"


# ============================================================
# INTERFACCIA GRADIO
# ============================================================

with gr.Blocks(
    title="Chatbot ARIA - Comune di Arezzo",
    theme="soft",
) as app:

    gr.Markdown("# 🟦 ARIA – Assistente del Comune di Arezzo")

    with gr.Row():
        question = gr.Textbox(
            label="Fai una domanda sul Comune di Arezzo",
            placeholder="Es: Quali sono gli orari degli uffici comunali?",
        )
        send_button = gr.Button("Invia")

    answer = gr.Markdown("Risposta…")
    status = gr.Markdown("⏳ Avvio…")

    # Chatbot con queue attiva
    send_button.click(
        fn=handle_question,
        inputs=question,
        outputs=answer,
        queue=True,
    )

    # Loader che non usa la queue
    app.load(
        fn=start_index_once,
        inputs=None,
        outputs=status,
        every=2,
        queue=False,
    )

app.queue(concurrency_count=3)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
