import streamlit as st
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from groq import Groq

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
COLLECTION_NAME = "rag_tp"
TOP_K = 3
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
LLM_MODEL = "llama-3.3-70b-versatile"

# Initialize clients
@st.cache_resource
def init_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def init_qdrant():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

@st.cache_resource
def init_groq():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Streamlit UI
st.title("📚 RAG Chatbot - TP2")
st.markdown("Pose une question sur ton document")

uploaded_file = st.file_uploader("📄 Charge ton document PDF", type="pdf")

if uploaded_file:
    # Read PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Chunk and index
    chunks = chunk_text(text)
    embedder = init_embedder()
    qdrant = init_qdrant()
    
    with st.spinner(f"📊 Indexation de {len(chunks)} chunks..."):
        embeddings = embedder.encode(chunks)
        
        # Reset collection
        try:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
        except:
            pass
        
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={"text": chunk}
            ))
        
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
    st.success(f"✅ Document indexé! {len(chunks)} chunks prêts.")
    
    # Question input
    question = st.text_input("💬 Ta question :")
    
    if st.button("Envoyer") and question:
        with st.spinner("🔍 Recherche et génération de réponse..."):
            # Retrieve
            q_emb = embedder.encode([question])[0]
            results = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=q_emb.tolist(),
                limit=TOP_K,
                with_payload=True
            )
            
            context = [point.payload["text"] for point in results.points]
            context_text = "\n\n".join(context)
            
            # Generate
            groq_client = init_groq()
            prompt = f"""Tu es un assistant. Réponds uniquement avec le contexte.

Contexte:
{context_text}

Question:
{question}"""
            
            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            
            st.write("### 🤖 Réponse :")
            st.write(response.choices[0].message.content)
            
            with st.expander("📖 Voir les sources"):
                for i, chunk in enumerate(context):
                    st.write(f"**Source {i+1}:**")
                    st.write(chunk)
