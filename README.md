# ai_agent_example
# Build a simple AI Agent for PDF Question Answering 📚 

This is a simple AI-powered web app built using **LangChain**, **Streamlit**, and **DeepSeek** that allows users to upload a PDF and ask questions about its content.

## 🚀 Features

- Upload and parse PDF documents
- Generate semantic embeddings using HuggingFace Transformers
- Perform similarity search with FAISS
- Use DeepSeek (or any LLM) to answer questions from extracted document chunks
- Interactive frontend built with Streamlit

## 🧠 Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Transformers
- FAISS
- DeepSeek LLM API
- PyPDF2

## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/mquazi/ai_agent_example.git
cd ai_agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

