import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import requests
import time
from typing import Tuple, List

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

INDEX_FILE = "tourisme.index"
CHUNKS_FILE = "chunks.pkl"

OLLAMA_PARAMS = {
    "temperature": 0.25,
    "top_p": 0.92,
    "top_k": 40,
    "stream": False
}

DEFAULT_K = 4

# ─────────────────────────────────────────────
# CSS pour un look Messenger moderne
# ─────────────────────────────────────────────
def apply_messenger_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f5;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px 10px;
        }
        .message {
            margin: 12px 0;
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #0084ff;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .assistant-message {
            background-color: white;
            color: #111;
            margin-right: auto;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .timestamp {
            font-size: 11px;
            color: #65676b;
            margin-top: 4px;
            text-align: right;
        }
        .stTextInput > div > div > input {
            border-radius: 24px !important;
            padding: 14px 20px !important;
            background-color: white !important;
            border: 1px solid #ddd !important;
        }
        .stButton > button {
            background-color: #0084ff !important;
            color: white !important;
            border-radius: 20px !important;
            padding: 10px 24px !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARGEMENT RESSOURCES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement...")
def load_resources() -> Tuple[SentenceTransformer, faiss.IndexFlatL2, List[str]]:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    if len(chunks) != index.ntotal:
        st.error("Incohérence index/chunks")
        st.stop()
    return embedder, index, chunks

# ─────────────────────────────────────────────
# APPEL OLLAMA
# ─────────────────────────────────────────────
def generate_with_ollama(prompt: str) -> Tuple[str, float]:
    try:
        start = time.time()
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, **OLLAMA_PARAMS},
            timeout=90
        )
        resp.raise_for_status()
        elapsed = time.time() - start
        return resp.json().get("response", "").strip(), elapsed
    except Exception as e:
        return f"Erreur : {str(e)}", 0.0

# ─────────────────────────────────────────────
# MAIN - Style Messenger
# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Voyagez en Tunisie", page_icon="🌍", layout="centered")
    apply_messenger_style()

    st.markdown("<h2 style='text-align:center; color:#0084ff; margin-bottom:1rem;'>Chat Tourisme & Culture</h2>", unsafe_allow_html=True)

    embedder, index, chunks = load_resources()

    # Initialisation historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage des messages comme un chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                st.markdown(f"""
                    <div class="message user-message">
                        {content}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="message assistant-message">
                        {content}
                    </div>
                """, unsafe_allow_html=True)

    # Saisie question
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([8, 2])
        with cols[0]:
            question = st.text_input(
                "Votre question...",
                placeholder="Ex. : Que voir à Tunis ?",
                label_visibility="collapsed"
            )
        with cols[1]:
            submit = st.form_submit_button("Envoyer")

    if submit and question.strip():
        # Ajouter question utilisateur
        st.session_state.messages.append({"role": "user", "content": question})

        # Préparer contexte RAG
        query_vec = embedder.encode([question]).astype("float32")
        _, indices = index.search(query_vec, k=DEFAULT_K)
        relevant_chunks = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
        context = "\n\n".join(relevant_chunks)

        prompt = f"""Tu es un guide touristique et culturel français.
Réponds UNIQUEMENT avec les informations du contexte.
Si rien ne correspond, dis simplement "Je ne sais pas".

*Contexte :*
{context}

*Question :*
{question}

*Réponse :* (naturelle, concise, amicale)"""

        # Génération
        with st.spinner("Réflexion..."):
            answer, _ = generate_with_ollama(prompt)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        # Rafraîchir
        st.rerun()

    # Bouton effacer
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

if _name_ == "_main_":
    main()