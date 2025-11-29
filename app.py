import os

# 🔥 Fix critici: elimina completamente audio + analytics
os.environ["GRADIO_DISABLE_ANALYTICS"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["GRADIO_AUDIO_DEPENDENCIES"] = "0"

import gradio as gr
import threading
import time

from src.rag_query import answer_question
from src.rag_pipeline import (
    build_index_if_needed,
    get_index_progress,
)

# ============================================================
# AVVIO INDICIZZAZIONE IN BACKGROUND
# ============================================================

index_started = False
index_ready = False

def ensure_index_built_async():
    """Richiama la pipeline solo una volta e costruisce l'indice FAISS."""
    global index_ready
    try:
        build_index_if_needed()
        index_ready = True
    except Exception as e:
        print("❌ Errore durante la costruzione indice:", e)
        index_ready = False


def start_index_once():
    """Lanciato da app.load() senza bloccare Gradio."""
    global index_started
    if not index_started:
        index_started = True
        threading.Thread(target=ensure_index_built_async, daemon=True).start()

    return progress_text()


# ============================================================
# PROGRESS BAR
# ============================================================

def progress_text():
    """Mostra stato percentuale e step dalla pipeline."""
    prog = get_index_progress()

    if prog["ready"]:
        return "📗 Indice pronto ✔️"

    percent = prog["percent"]
    step = prog["step"]

    return f"🟧 Preparazione indice… {percent}% (step: {step})"


# ============================================================
# FUNZIONE DI RISPOSTA RAG
# ============================================================

def handle_question(q: str):
    if not q.strip():
        return "Inserisci una domanda."

    if not index_ready:
        return "⏳ L'indice è ancora in preparazione, attendi qualche istante…"

    try:
        res = answer_question(q)

        # Se fosse un generatore evitiamo crash
        if hasattr(res, "__iter__") and not isinstance(res, str):
            last = None
            for x in res:  # Consuma il generatore
                last = x
            return last or "Errore sconosciuto."
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

    # Bot: Queue ON (Fondamentale!)
    send_button.click(
        fn=handle_question,
        inputs=question,
        outputs=answer,
        queue=True,
    )

    # Loader: Queue OFF
    app.load(
        fn=start_index_once,
        inputs=None,
        outputs=status,
        every=2,
        queue=False,
    )

# Queue attiva solo per la chat
app.queue(concurrency_count=3)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0",
               server_port=int(os.getenv("PORT", 7860)))
