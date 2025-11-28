
# Chatbot ARIA – Comune di Arezzo (Deploy su Render)

Applicazione Python con:
- Crawler del sito https://www.comune.arezzo.it
- Indicizzazione RAG con OpenAI
- Interfaccia web Gradio

## Avvio locale

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="..."
python app.py
```

## Deploy su Render

- Create un nuovo servizio Web collegato alla repo.
- Command: `python app.py`
- Environment: `OPENAI_API_KEY` con la vostra chiave OpenAI.
- (Opzionale) Aggiungete un disco persistente collegato alla cartella `/app/data`.
