# AgenticRAG 🤖

A sophisticated AI Agent application capable of answering user queries by intelligently leveraging **both a private knowledge base** (RAG) and **real-time web search**. Users have **granular control** over web search, enhancing flexibility, transparency, and trust in AI responses.

---

## ✨ Key Features

- **Hybrid AI & Intelligent Routing**: Combines internal RAG knowledge with real-time web search to select the best information source dynamically.  
- **User-Controlled Web Access**: Toggle web search on/off for internal-only knowledge or broader internet queries.  
- **Transparent AI Workflow (Agent Trace)**: Step-by-step trace of the agent’s thought process, routing decisions, and RAG sufficiency verdicts.  
- **Contextual RAG Sufficiency Judgment**: LLM evaluates if retrieved RAG content is enough to answer a query.  
- **Dynamic Knowledge Ingestion (PDF Upload)**: Upload PDFs directly; they are automatically processed, embedded, and added to Pinecone.  
- **Modular & Extensible Design**: Clean architecture using FastAPI, LangGraph, Streamlit for easy debugging and extension.  
- **Persistent Conversation Memory**: LangGraph checkpointing maintains context across multiple turns.

---

## 🚀 High-Level Architecture

**Layers Overview:**

| Layer                 | Description |
|-----------------------|-------------|
| **User Interface (UI)** | Streamlit app for interaction |
| **API Layer**           | FastAPI backend handling requests |
| **Agent Core**          | LangGraph-powered AI logic with routing and tools |
| **Knowledge Base**      | Pinecone vector DB + HuggingFace embeddings |
| **External Tools**      | Groq LLM, Tavily Search API |

---

## 📦 Core Modules Structure

agentBot/
├── frontend/
│   ├── app.py                  # Streamlit entry point
│   ├── ui_components.py       # Chat UI, toggle, trace
│   ├── backend_api.py         # API communication
│   ├── session_manager.py     # Streamlit state management
│   └── config.py              # Frontend config
│
├── backend/
│   ├── main.py                # FastAPI entry point
│   ├── agent.py               # LangGraph AI agent workflow
│   ├── vectorstore.py         # Pinecone RAG logic
│   └── config.py              # API keys and env vars
│
│
├── requirements.txt          # Python dependencies
└── .env                      # API keys (not committed)


---

## ⚙️ Technology Stack

- **Language:** Python 3.9+  
- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **Agent Orchestration:** LangGraph  
- **LLMs & Tools:** LangChain, Groq (Llama 3)  
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector Store:** Pinecone  
- **PDF Processing:** PyPDFLoader  
- **Search Engine:** Tavily API  

---

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.9+  
- API Keys: `GROQ_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY`  
- Pinecone index: `rag-index` with 384 dimensions and cosine metric  

### Steps

```bash
# Clone the repo
git clone https://github.com/your-username/agentBot.git
cd agentBot

# Create virtual environment
python -m venv .venv
# Activate it:
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file at project root
# Example:
GROQ_API_KEY="your_groq_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_ENVIRONMENT="your_pinecone_environment"
TAVILY_API_KEY="your_tavily_api_key"
FASTAPI_BASE_URL="http://localhost:8000"



cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

cd ..
streamlit run frontend/app.py

**### 🧪 API Testing with Postman**

Upload Document

POST /upload-document/

URL: http://localhost:8000/upload-document/
Body: form-data, key=file, type=File

Response:
{
  "message": "PDF 'doc.pdf' successfully uploaded and indexed.",
  "filename": "doc.pdf",
  "processed_chunks": 5
}

hat Query

POST /chat/

URL: http://localhost:8000/chat/
Body:{
  "session_id": "test-session-001",
  "query": "What are the treatments for diabetes?",
  "enable_web_search": true
}

**Response:**
{
  "response": "Your agent's answer here...",
  "trace_events": [
    {
      "step": 1,
      "node_name": "router",
      "description": "...",
      "event_type": "router_decision"
    }
  ]
}
