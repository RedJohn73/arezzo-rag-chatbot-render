import os

# 🔥 Fix critico: disattiva completamente l’audio in Gradio ed evita crash pyaudioop
os.environ["GRADIO_DISABLE_ANALYTICS"] = "1"
os.environ["GRADIO_AUDIO_DEPENDENCIES"] = "0"

import gradio as gr
from src.rag_pipeline import answer_question, ensure_index_built_async, get_progress
import threading

# -----------------------------
#  AVVIO INDICIZZAZIONE IN BACKGROUND
# -----------------------------
def start_index():
    try:
        print("[INFO] Starting background index build...")
        ensure_index_built_async()
    except Exception as e:
        print("[ERROR] Index build failed:", e)

threading.Thread(target=start_index, daemon=True).start()


# -----------------------------
#  FUNZIONI UI
# -----------------------------
def query_bot(user_input):
    if not user_input.strip():
        return "Inserisci una domanda."
    return answer_question(user_input)


# Indicatore STATICO — niente generatori, niente refresh automatici
def get_status_label():
    prog = get_progress()
    if prog["ready"]:
        return f"📦 Indice pronto – {prog['percent']}%"
    return f"📡 Preparazione indice… {prog['percent']}%"


# -----------------------------
#  INTERFACCIA GRADIO
# -----------------------------
with gr.Blocks(title="Chatbot ARIA - Comune di Arezzo") as app:

    gr.Markdown("# 🟦 ARIA – Assistente del Comune di Arezzo")

    with gr.Row():
        question = gr.Textbox(
            label="Fai una domanda sul Comune di Arezzo",
            placeholder="Es: Quali sono gli orari degli uffici comunali?",
        )
        ask_btn = gr.Button("Invia")

    answer = gr.Markdown("Risposta…")

    # 🔵 Indicatore statico e stabile
    status = gr.Markdown(get_status_label())

    # Azione del bottone
    ask_btn.click(fn=query_bot, inputs=question, outputs=answer)


app.queue()  # necessari per stabilità su Gradio 4.x

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))

