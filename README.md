# Designing Multi-Agent Autonomous Systems with Microsoft Azure OpenAI

# 🧠 Streamlit RAG Application (with Query Rewriter Agent)

This repository contains a **Streamlit**-based **Retrieval-Augmented Generation (RAG)** application built with:
- **OpenAI's LLMs**
- **Qdrant Cloud** (for vector storage and retrieval)
- **Agno AI Agents** (for query rewriting)
- **LangChain** (for document loading and chunking)

> 📄 Currently, **only PDF files** are supported for document ingestion.

---

## 📂 Project Structure

| File | Description |
|:-----|:------------|
| `basic_rag.py` | Basic RAG system using OpenAI Embedding (`models/text-embedding-004`), OpenAI LLM (`gpt4o`), and Qdrant Cloud as the vector database. |
| `agentic_rag.py` | Extends `basic_rag.py` by adding a **Query Rewriter Agent** powered by **Agno AI**, improving retrieval relevance. |
| `agentic_rag.py` | Extends `basic_rag.py` by adding a **Exa Search Tool** again powered by **Agno AI** to search content over web in case not available in RAG |
| `requirements.txt` | Python dependencies list to install all necessary packages. |

---

## 🚀 How to Run the Application

1. **Clone the repository**:
    ```bash
    git clone https://github.com/explore-with-yasir/rag_ai_agents_coworkers.git
    cd rag_ai_agents_coworkers
    ```

2. **Set up a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Create a `.env` file** in the root directory and add the following API keys:
    ```env
    AZURE_OPENAI_API_KEY=your-openai-api-key
    QDRANT_API_KEY=your-qdrant-api-key
    QDRANT_URL=https://your-qdrant-cloud-instance-url
    EXA_API_KEY=your-exa-api-key  # Required only for basic_rag_with_rewriter.py
    ```

5. **Run the Streamlit app**:
    - For basic RAG:
      ```bash
      streamlit run basic_rag.py
      ```
    - For RAG with query rewriter:
      ```bash
      streamlit run basic_rag_with_rewriter.py
      ```

---

## 🔑 How to Get API Keys

### 1. OpenAI API Key
- Go to [Open AI](https://platform.openai.com/api-keys).
- Sign in with your OpenAI account.
- Create a new API key for accessing OpenAI models.

### 2. Qdrant Cloud API Key
- Sign up at [Qdrant Cloud](https://cloud.qdrant.io/).
- Create a new cluster (select region, resources).
- Go to **Settings** → **API Keys** and generate a key.
- Note the **Cluster URL** (you will set this as `QDRANT_URL`) and the key (`QDRANT_API_KEY`).

### 3. Exa Search API Key
- Create an account at [Exa (by Exa Labs)](https://exa.ai/).
- Navigate to your account settings and generate an API key.

---

## 💬 What is Streamlit?

[Streamlit](https://streamlit.io/) is an open-source Python framework that allows you to build interactive web apps for machine learning and data science projects quickly — just by writing Python scripts.

In this project, **Streamlit** powers the user interface for:
- Uploading PDFs
- Asking questions
- Viewing the retrieved and generated responses

---

## ⚙️ Libraries and Technologies Used

- **Streamlit**: Web UI
- **OpenAI Generative AI (OpenAI)**: 
  - Embedding model: `models/text-embedding-004`
  - LLM model: `gpt4o`
- **Qdrant Cloud**: Vector Database
- **LangChain**: PDF loading, chunking, and integration with Qdrant
- **Agno AI**: 
  - Agent framework to manage a **Query Rewriter Agent**
  - Integrated tools like **Exa Search**
- **Exa Tools**: Web search fallback to improve document retrieval

### Python Libraries Imported
```python
import os
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
from agno.agent import Agent
from agno.tools.exa import ExaTools
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
```

---

## 📚 Notes

- 📄 Currently **only PDF uploads** are supported for ingestion.
- 🛠️ Qdrant vectors are indexed using **Cosine Distance**.
- 🚀 The agentic version (`basic_rag_with_rewriter.py`) improves queries using web search context when necessary.
- ⚡ The application uses **OpenAI Embedding** model for document chunk embeddings and **GPT4o** model for answering.

---

## ✨ Future Improvements

- Support other document formats (e.g., websites, DOCX).
- Add multi-file support.
- Enhance agent reasoning with custom tools.

---

## ✅ Written by [Yasir Siddique](https://www.yasirsiddique.com). My [Linkedin](https://www.linkedin.com/in/yasir-sd)

