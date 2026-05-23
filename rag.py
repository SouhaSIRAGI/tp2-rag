import os
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# =========================
# CONFIG
# =========================

EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "rag_tp"

TOP_K = 3

# 🔑 Put your API key here OR use environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY")

LLM_MODEL = "llama-3.3-70b-versatile"

# =========================
# LOAD MODELS
# =========================

embedder = SentenceTransformer(EMBEDDING_MODEL)

client = Groq(api_key=GROQ_API_KEY)

# Qdrant local or cloud (you can replace later)
qdrant = QdrantClient(":memory:")

# =========================
# RETRIEVAL FUNCTION
# =========================

def retrieve(query: str):
    """Search most relevant chunks from vector DB"""

    query_vector = embedder.encode(query)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=TOP_K,
        with_payload=True
    )

    return [r.payload["text"] for r in results]


# =========================
# RAG PIPELINE
# =========================

def rag(question: str) -> str:
    """Full RAG pipeline: retrieve + generate"""

    # 1. Retrieve context
    context_chunks = retrieve(question)

    context = "\n\n".join(context_chunks)

    # 2. Build prompt
    prompt = f"""
Tu es un assistant intelligent.

Réponds UNIQUEMENT avec le contexte ci-dessous.
Si l'information n'existe pas, dis "Je ne trouve pas cette information".

Contexte :
{context}

Question :
{question}
"""

    # 3. Call LLM (Groq)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )

    return response.choices[0].message.content
