import os
import tempfile
import warnings
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader

from backend.agent import rag_agent
from backend.vectorstore import add_document_to_vectorstore

# --- Suppress LangChain Deprecation Warnings ---
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

app = FastAPI(
    title="LangGraph RAG Agent API",
    description="API for LangGraph-powered RAG agent with Pinecone and Groq.",
    version="1.0.0",
)

# --- Pydantic Models ---
class TraceEvent(BaseModel):
    step: int
    node_name: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    event_type: str

class QueryRequest(BaseModel):
    session_id: str
    query: str
    enable_web_search: bool = True

class AgentResponse(BaseModel):
    response: str
    trace_events: List[TraceEvent] = Field(default_factory=list)

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int


# --- Document Upload ---
@app.post("/upload-document/", response_model=DocumentUploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        file_content = await file.read()
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # Load the PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        if not documents:
            return DocumentUploadResponse(
                message=f"PDF '{file.filename}' has no content to index.",
                filename=file.filename,
                processed_chunks=0,
            )

        # Combine all pages into a single string
        full_text_content = "\n\n".join([doc.page_content for doc in documents])

        # Add to vectorstore (will auto-create index if missing and delete old chunks)
        add_document_to_vectorstore(full_text_content, file_name=file.filename)

        # Count chunks (using the text splitter)
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        chunks = text_splitter.split_text(full_text_content)
        total_chunks_added = len(chunks)

        return DocumentUploadResponse(
            message=f"PDF '{file.filename}' successfully uploaded and indexed.",
            filename=file.filename,
            processed_chunks=total_chunks_added,
        )

    finally:
        # Clean up temp file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass


# --- Chat Endpoint ---
@app.post("/chat/", response_model=AgentResponse)
async def chat_with_agent(request: QueryRequest):
    trace_events: List[TraceEvent] = []
    node_output_state: Dict[str, Any] = {}

    config = {"configurable": {"thread_id": request.session_id, "web_search_enabled": request.enable_web_search}}
    inputs = {"messages": [HumanMessage(content=request.query)]}
    final_message = ""

    try:
        for i, s in enumerate(rag_agent.stream(inputs, config)):
            if "__end__" in s:
                current_node_name = "__end__"
                node_output_state = s["__end__"]
            else:
                keys = list(s.keys())
                if not keys:
                    continue
                current_node_name = keys[0]
                node_output_state = s[current_node_name]

            trace_events.append(
                TraceEvent(
                    step=i + 1,
                    node_name=current_node_name,
                    description=f"Node executed: {current_node_name}",
                    details={},
                    event_type="execution",
                )
            )

        # get final AI message
        if node_output_state and "messages" in node_output_state:
            for msg in reversed(node_output_state["messages"]):
                if isinstance(msg, AIMessage):
                    final_message = msg.content
                    break

        if not final_message:
            raise HTTPException(status_code=500, detail="Agent did not return a response")

        return AgentResponse(response=final_message, trace_events=trace_events)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
