import os
import json
from typing import TypedDict, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field, ValidationError
from backend.config import GROQ_API_KEY, TAVILY_API_KEY
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from backend.vectorstore import get_retriever
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- Tools ---
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily = TavilySearch(max_results=3, topic="general")


@tool(description="Up-to-date web info via Tavily")
def web_search_tool(query: str) -> str:
    """Return top web search results for a query using Tavily."""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and "results" in result:
            formatted_results = [
                f"Title: {item.get('title','No title')}\nContent: {item.get('content','No content')}\nURL: {item.get('url','')}"
                for item in result["results"]
            ]
            return "\n\n".join(formatted_results) or "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"


@tool(description="Top-K chunks from KB (RAG) retrieval")
def rag_search_tool(query: str) -> str:
    """Fetch top chunks from knowledge base using RAG."""
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query, k=5)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"


# --- Pydantic schemas ---
class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None)


class RagJudge(BaseModel):
    sufficient: bool = Field(...)


# --- Safe parse functions ---
def safe_parse_route(resp) -> dict:
    """Ensure we always return dict, even if LLM response fails."""
    if isinstance(resp, RouteDecision):
        return resp.dict()
    try:
        return RouteDecision(**json.loads(resp)).dict()
    except (json.JSONDecodeError, ValidationError, TypeError):
        return RouteDecision(route="rag").dict()


def safe_parse_judge(resp) -> dict:
    """Ensure we always return dict, even if LLM response fails."""
    if isinstance(resp, RagJudge):
        return resp.dict()
    try:
        return RagJudge(**json.loads(resp)).dict()
    except (json.JSONDecodeError, ValidationError, TypeError):
        return RagJudge(sufficient=False).dict()


# --- LLM instances ---
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
router_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY).with_structured_output(RagJudge)
answer_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.7)


# --- Shared state ---
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str
    web_search_enabled: bool


# --- Nodes ---
def router_node(state: AgentState, config) -> AgentState:
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)

    messages = [
        ("system", "You are a routing agent. Decide best route for query."),
        ("user", query)
    ]

    raw_resp = router_llm.invoke(messages)
    result = safe_parse_route(raw_resp)

    if not web_search_enabled and result["route"] == "web":
        result["route"] = "rag"

    out = {
        "messages": state["messages"],
        "route": result["route"],
        "web_search_enabled": web_search_enabled
    }
    if result["route"] == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.get("reply") or "Hello!")]
    return out


def rag_node(state: AgentState, config) -> AgentState:
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)

    chunks = rag_search_tool.invoke(query)
    if chunks.startswith("RAG_ERROR::"):
        return {**state, "rag": "", "route": "web" if web_search_enabled else "answer"}

    judge_messages = [
        ("system", "Evaluate if retrieved info is sufficient. Respond ONLY as JSON {\"sufficient\": true/false}"),
        ("user", f"Question: {query}\nRetrieved info: {chunks}")
    ]
    raw_verdict = judge_llm.invoke(judge_messages)
    verdict = safe_parse_judge(raw_verdict)

    next_route = "answer" if verdict["sufficient"] else ("web" if web_search_enabled else "answer")
    return {**state, "rag": chunks, "route": next_route, "web_search_enabled": web_search_enabled}


def web_node(state: AgentState, config) -> AgentState:
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)

    if not web_search_enabled:
        return {**state, "web": "Web search disabled", "route": "answer"}

    snippets = web_search_tool.invoke(query)
    if snippets.startswith("WEB_ERROR::"):
        return {**state, "web": "", "route": "answer"}

    return {**state, "web": snippets, "route": "answer"}


def answer_node(state: AgentState) -> AgentState:
    user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    ctx_parts = []

    if state.get("rag"):
        ctx_parts.append("Knowledge Base:\n" + state["rag"])
    if state.get("web") and not state["web"].startswith("Web search disabled"):
        ctx_parts.append("Web Search Results:\n" + state["web"])

    context = "\n\n".join(ctx_parts) or "No context available, answer based on general knowledge."
    prompt = f"Answer the question using context.\n\nQuestion: {user_q}\n\nContext:\n{context}"
    ans = answer_llm.invoke([HumanMessage(content=prompt)]).content

    return {**state, "messages": state["messages"] + [AIMessage(content=ans)]}


# --- Routing helpers ---
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]


def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]


def after_web(_) -> Literal["answer"]:
    return "answer"


# --- Build agent ---
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("rag_lookup", rag_node)
    g.add_node("web_search", web_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        from_router,
        {"rag": "rag_lookup", "web": "web_search", "answer": "answer", "end": END},
    )
    g.add_conditional_edges("rag_lookup", after_rag, {"answer": "answer", "web": "web_search"})
    g.add_edge("web_search", "answer")
    g.add_edge("answer", END)

    return g.compile(checkpointer=MemorySaver())


rag_agent = build_agent()
