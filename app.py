

import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

# API configuration for Deepseek
DEEPSEEK_API_KEY = ""  
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# Configure webpage
st.set_page_config(page_title="Ask a question on the PDF/Document")
st.header("Ask questions on your PDF/Document 📝 📚")

# Configure file upload
pdf = st.file_uploader("📄 Upload your document", type="pdf")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process the PDF and create a vector db
if pdf is not None:
    # Read PDF
    pdf_reader = PdfReader(pdf)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
        #length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
    st.success("The document has been processed successfully!")

# QnA configuration
if st.session_state.vector_store is not None:
    question = st.text_area(
        "👩‍💻 Interact with your PDF:", 
        height=100,  
        max_chars=500,  
        help="Type your question here..."  
    )
    if question:
        # Search for similar chunks
        docs = st.session_state.vector_store.similarity_search(question)
        # Initialize Deepseek LLM
        llm = ChatOpenAI(
            model_name="deepseek-chat",
            temperature=0,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
            streaming=False
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.invoke({"input_documents": docs, "question": question})["output_text"]
        st.markdown(":bulb: **Answer:**")  
        st.write(response)
