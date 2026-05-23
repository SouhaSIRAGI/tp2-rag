import streamlit as st
from rag import rag

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📚 RAG Chatbot - TP2")

st.write("Pose une question sur ton document")

# Input user
question = st.text_input("Ta question :")

# Button
if st.button("Envoyer"):

    if question.strip() != "":
        with st.spinner("Réflexion en cours... 🤖"):

            answer = rag(question)

        st.success("Réponse :")
        st.write(answer)

    else:
        st.warning("Veuillez entrer une question")
