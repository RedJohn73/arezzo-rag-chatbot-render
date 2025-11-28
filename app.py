
import os
import threading

import gradio as gr
from dotenv import load_dotenv

from src.rag_pipeline import answer_question, ensure_index_built_async, get_progress

load_dotenv()

# Avvia la costruzione/aggiornamento dell'indice in background
def start_background_indexer():
    t = threading.Thread(target=ensure_index_built_async, daemon=True)
    t.start()

with gr.Blocks(title="Chatbot ARIA - Comune di Arezzo") as demo:
    gr.Markdown("# 🏛️ Chatbot ARIA – Comune di Arezzo")
    gr.Markdown(
        """Assistente istituzionale basato su:
        - Crawling del sito ufficiale del Comune di Arezzo
        - Indicizzazione RAG con OpenAI
        """
    )

    with gr.Row():
        status_box = gr.Textbox(
            label="Stato indicizzazione",
            interactive=False,
            value="Inizializzazione..."
        )
        refresh_btn = gr.Button("🔄 Aggiorna stato")

    def ui_get_progress():
        p = get_progress()
        return f"{p['percent']}% – step: {p['step']} – pronto: {p['ready']}"

    refresh_btn.click(fn=ui_get_progress, inputs=None, outputs=status_box)

    chat_input = gr.Textbox(
        label="Domanda",
        placeholder="Scrivi una domanda sui servizi del Comune di Arezzo...",
        lines=2,
    )
    chat_output = gr.Textbox(label="Risposta", lines=10)

    def chat_fn(message):
        return answer_question(message)

    send_btn = gr.Button("Invia")
    send_btn.click(fn=chat_fn, inputs=chat_input, outputs=chat_output)
    chat_input.submit(fn=chat_fn, inputs=chat_input, outputs=chat_output)

if __name__ == "__main__":
    # Avvia l'indicizzazione in background e poi il server Gradio
    start_background_indexer()
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
