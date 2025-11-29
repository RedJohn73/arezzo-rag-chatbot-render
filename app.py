import os

# Disabilita problemi noti di Gradio
os.environ["GRADIO_DISABLE_ANALYTICS"] = "1"
os.environ["GRADIO_AUDIO_DEPENDENCIES"] = "0"

import gradio as gr
from src.rag_pipeline import answer_question, ensure_index_built_async, get_progress

# ---------------------------------------------------------------------
# Stato indicizzazione
# ---------------------------------------------------------------------
_index_started = False

def start_index_if_needed():
    global _index_started
    if not _index_started:
        print("[INFO] Starting index build…")
        _index_started = True
        ensure_index_built_async()
    return progress_status()

def progress_status():
    prog = get_progress()
    if prog["ready"]:
        return "Indice pronto ✔️"
    return f"🕗 Preparazione indice… {prog['percent']}%"

def query_bot(user_input):
    if not user_input.strip():
        return "Inserisci una domanda."
    return answer_question(user_input)

# ---------------------------------------------------------------------
# UI Gradio
# ---------------------------------------------------------------------
with gr.Blocks(title="Chatbot ARIA - Comune di Arezzo") as app:

    gr.Markdown("# 🟦 ARIA – Assistente del Comune di Arezzo")

    with gr.Row():
        question = gr.Textbox(
            label="Fai una domanda sul Comune di Arezzo",
            placeholder="Es: Quali sono gli orari degli uffici comunali?",
        )
        ask_btn = gr.Button("Invia")

    answer = gr.Markdown("Risposta…")
    status = gr.Markdown("🔄 Avvio…")

    ask_btn.click(
        fn=query_bot,
        inputs=question,
        outputs=answer,
        queue=True      # 👈 NECESSARIO per funzioni bot lente/LLM
    )

    app.load(
        fn=start_index_if_needed,
        inputs=None,
        outputs=status,
        every=3,
        queue=False      # 👈 IMPORTANTE: loader NON va in queue
    )

# Attiva la queue solo per le chiamate degli utenti
app.queue(concurrency_count=8, max_size=32)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
