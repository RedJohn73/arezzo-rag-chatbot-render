import json
import numpy as np
import faiss
from openai import OpenAI
import os

INDEX_DIR = "vectorstore"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "meta.json")

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1-mini"

client = OpenAI()  # prende la OPENAI_API_KEY automaticamente


def load_faiss():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise RuntimeError("Indice FAISS mancante. Esegui prima la pipeline.")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r") as f:
        meta = json.load(f)
    return index, meta


def embed_query(q):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=q
    )
    return np.array(resp.data[0].embedding, dtype="float32")


def search_similar(query, k=3):
    index, meta = load_faiss()
    v = embed_query(query)
    v = np.expand_dims(v, axis=0)
    distances, indices = index.search(v, k)

    results = []
    for idx in indices[0]:
        if idx < len(meta):
            results.append(meta[idx])
    return results


def answer_question(query):
    docs = search_similar(query)

    context = "\n\n".join([d["text"] for d in docs])

    prompt = f"""
Sei ARIA, assistente istituzionale del Comune di Arezzo.
Rispondi in modo accurato e ufficiale usando SOLO le informazioni seguenti.

Contesto:
{context}

Domanda: {query}

Risposta istituzionale:
"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Rispondi in modo istituzionale e preciso."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message["content"]
