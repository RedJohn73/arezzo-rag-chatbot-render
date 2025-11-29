import os
# 🔥 Fix critico: disattiva completamente l’audio in Gradio e impedisce il crash pyaudioop
os.environ["GRADIO_DISABLE_ANALYTICS"] = "1"
os.environ["GRADIO_AUDIO_DEPENDENCIES"] = "0"

import gradio as gr
from src.rag_pipeline import answer_question, ensure_index_built_async, get_progress
import threading

# Avvia Indicizzazione in background
def start_index():
    try:
        print("[INFO] Starting background index build…")
        ensure_index_built_async()
    except Exception as e:
        print("[ERROR] Index build failed:", e)

threading.Thread(target=start_index, daemon=True).start()

# Interfaccia Utente
def query_bot(user_input):
    if not user_input.strip():
        return "Inserisci una domanda."
    return answer_question(user_input)

def progress_status():
    prog = get_progress()
    if prog["ready"]:
        return "Indice pronto ✔️"
    return f"Indicizzazione in corso… {prog['percent']}%"

with gr.Blocks(title="Chatbot ARIA - Comune di Arezzo") as app:
    gr.Markdown("# 🟦 ARIA – Assistente del Comune di Arezzo")

    with gr.Row():
        question = gr.Textbox(
            label="Fai una domanda sul Comune di Arezzo",
            placeholder="Es: Quali sono gli orari degli uffici comunali?",
        )
        ask_btn = gr.Button("Invia")

    answer = gr.Markdown("Risposta…")

    status = gr.Markdown("📡 Stato: avvio in corso…")

    ask_btn.click(fn=query_bot, inputs=question, outputs=answer)
    app.load(fn=progress_status, inputs=None, outputs=status, every=3)

app.queue()   # ⭐ NECESSARIO PER FUNZIONAMENTO SU GRADIO 3.x

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))

