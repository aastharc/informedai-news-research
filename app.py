import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

from langchain.vectorstores import FAISS

from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login

import subprocess
import streamlit as st






# Load environment variables


# Set page configuration
st.set_page_config(page_title="InformedAI", layout="centered")
AUTH = st.secrets["HUGGING_FACE_AUTH"]
login(AUTH)
# Page title
st.markdown("<h1 style='text-align: center;'> InformedAI: News Research Tool 📈</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: gray;'>Your intelligent news assistant, powered by language models and real-world data.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px; color: gray;'>Great for searching through long articles you can’t feed into ChatGPT due to prompt limits.</p>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center;'>
        <h3>🛠️ How to Use:</h3>
        <ol style='text-align: left; display: inline-block; font-size: 16px;'>
            <li>🔗 Enter the latest articles/ URLs of your choice in the sidebar. Click Process.</li>
            <li>❓ Ask any question based on the content of those articles and press enter.</li>
            <li>📚 Get answers along with source references from the provided links.</li>
        </ol>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
    /* Reduce padding inside sidebar container */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    /* Collapse spacing around input and radio buttons */
    div[data-baseweb="input"] {
        margin-bottom: 0rem !important;
    }
    div[data-testid="stRadio"] {
        margin-top: 0rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Tighten spacing for markdown titles and dividers */
    .stMarkdown h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }

    hr {
        margin: 0.2rem 0 !important;
    }
    </style>
""", unsafe_allow_html=True)
# Sidebar for URL input
st.sidebar.header("🔗 Enter News Article URLs or just paste the article")
input_areas = []

for i in range(4):
    st.sidebar.markdown(f"**Input {i+1}**")
    
    # Default value container
    #input_value = st.sidebar.text_input(f"", key=f"input_{i}")

    # Input type (URL / Paste) – horizontal radio buttons
    cols = st.sidebar.columns([3, 2]) 
    with cols[0]:
        input_value = st.text_input("", key=f"input_{i}", label_visibility="collapsed")
    with cols[1]:
        input_type = st.sidebar.radio(
            "",
            ("URL", "Paste"),
            horizontal=True,
            key=f"type_{i}"
        )

    if input_value.strip():
        input_areas.append({
            "type": "url" if input_type == "URL" else "text",
            "value": input_value.strip()
        })
    if i != 3:
        st.sidebar.markdown("---", unsafe_allow_html=True)




process_inputs_clicked= st.sidebar.button("📥 Process")

# Model setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llmi = HuggingFacePipeline(pipeline=pipe)
vectorindex_huggingface = None
main_placeholder = st.empty()
if process_inputs_clicked:
    urls = [entry["value"] for entry in input_areas if entry["type"] == "url"]
    texts = [entry["value"] for entry in input_areas if entry["type"] == "text"]

    data = []

    if urls:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("🔗 Loading URL articles...")
        data += loader.load()

    if texts:
        # Convert pasted articles to LangChain-style documents
        for text in texts:
            data.append({"page_content": text})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.split_documents(data)

    main_placeholder.text("📚 Splitting text into chunks...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorindex_huggingface = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("📦 Creating Embedding Vector Index... Done!")
    time.sleep(2)

    

# Question input
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("💬 Ask a Question Based on the News")
query = st.text_input("Type your question here:")

# Show answer
if query and vectorindex_huggingface is not None:
    vectorstore = vectorindex_huggingface

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llmi, retriever=vectorstore.as_retriever())
    with st.spinner("🤖 Generating answer..."):
        response = chain({"question": query}, return_only_outputs=True)

    st.markdown("### 🧠 Answer")
    st.success(response["answer"])

    sources = response.get("sources", "")
    if sources:
        st.markdown("### 🔎 Sources")
        for source in sources.split("\n"):
            st.markdown(f"- {source}")
elif query:
    st.warning("⚠️ Please click '📥 Process' after entering URLs/ articles.")