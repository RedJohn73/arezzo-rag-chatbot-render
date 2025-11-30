import os

# 🔥 Fix critici: elimina completamente audio + analytics
os.environ["GRADIO_DISABLE_ANALYTICS"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["GRADIO_AUDIO_DEPENDENCIES"] = "0"

import gradio as gr
import threading

from src.rag_query import answer_question
from src.rag_pipeline import index_exists, build_index_async, get_progress

# ============================================================
# AVVIO INDICIZZAZIONE IN BACKGROUND
# ============================================================

index_started = False

def start_index_once():
    """
    Avvia la costruzione dell'indice solo la prima volta.
    """
    global index_started
    if not index_started:
        index_started = True
        threading.Thread(target=build_index_async, daemon=True).start()
    return progress_text()


def progress_text():
    """
    Testo dello stato dell'indicizzazione per Gradio.
    """
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

        # Se per sbaglio restituisce un generatore → prendiamo l'ultimo valore
        if hasattr(res, "__iter__") and not isinstance(res, str):
            last = None
            for x in res:
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

    # Bot: Queue ON
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

app.queue(concurrency_count=3)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860))
    )
