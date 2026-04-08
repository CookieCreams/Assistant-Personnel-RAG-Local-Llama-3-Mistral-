import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Mon IA Perso", page_icon="🤖", layout="wide")
st.title("🤖 Assistant Personnel Local")

# --- INITIALISATION DES MOTEURS ---
@st.cache_resource
def init_models():
    llm_model   = os.getenv("LLM_MODEL", "mistral")
    embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    embeddings = OllamaEmbeddings(model=embed_model)
    llm = ChatOllama(model=llm_model, temperature=0.3)
    db = Chroma(persist_directory="./ma_base", embedding_function=embeddings)
    return llm, embeddings, db

llm, embeddings, db = init_models()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("📁 Administration")

    uploaded_file = st.file_uploader("Ajouter un PDF à la base", type="pdf")

    if uploaded_file is not None:
        if st.button("🚀 Indexer définitivement"):
            with st.status("Analyse et stockage..."):
                if not os.path.exists("./data"):
                    os.makedirs("./data")
                path = os.path.join("./data", uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader(path)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, chunk_overlap=80
                )
                chunks = text_splitter.split_documents(loader.load())
                db.add_documents(chunks)

                # FIX : on force le rechargement de la ressource cachée
                init_models.clear()

            st.success(f"'{uploaded_file.name}' ajouté à la base !")

    st.write("---")

    # Affichage du nombre de documents indexés
    try:
        count = db._collection.count()
        st.metric("Chunks indexés", count)
    except Exception:
        pass

    st.write("---")
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

# --- MÉMOIRE DE SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- AFFICHAGE DE L'HISTORIQUE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ZONE DE CHAT ---
if prompt := st.chat_input("Posez-moi une question..."):

    # 1. Affichage et sauvegarde du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Réponse de l'assistant
    with st.chat_message("assistant"):

        # --- Recherche RAG dans la base ---
        docs = db.similarity_search(prompt, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # --- Construction des messages avec historique ---
        # Le LLM reçoit maintenant TOUT l'historique, pas juste la dernière question
        system_message = SystemMessage(content=f"""Tu es un assistant personnel intelligent et précis.
Utilise le contexte documentaire ci-dessous pour répondre aux questions.
Si la réponse ne se trouve pas dans le contexte, utilise tes connaissances générales en le signalant clairement.
Réponds toujours en français, de façon structurée et concise.

--- CONTEXTE DOCUMENTAIRE ---
{context}
--- FIN DU CONTEXTE ---""")

        # Reconstitution de l'historique au format LangChain
        history = []
        for msg in st.session_state.messages[:-1]:  # Tout sauf le dernier (déjà ajouté)
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))

        # Message actuel
        current_message = HumanMessage(content=prompt)

        messages_to_send = [system_message] + history + [current_message]

        # --- Streaming de la réponse ---
        response_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(messages_to_send):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")  # curseur animé

        response_placeholder.markdown(full_response)  # Affichage final propre

        # --- Affichage des sources utilisées ---
        if docs:
            with st.expander("📚 Sources utilisées", expanded=False):
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "Document inconnu")
                    page = doc.metadata.get("page", "?")
                    st.caption(f"**[{i+1}] {os.path.basename(source)} — page {page}**")
                    st.text(doc.page_content[:300] + "...")

    # 3. Sauvegarde de la réponse
    st.session_state.messages.append({"role": "assistant", "content": full_response})