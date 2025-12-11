'''
from typing import TypedDict, Literal, Dict, Any

import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch

from qdrant_utils import get_qdrant_client, get_embedder

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

QNA_COLLECTION = "medical_qna"
DEVICE_COLLECTION = "medical_device_manual"


class GraphState(TypedDict):
    query: str
    context: str
    prompt: str
    response: str
    source: str
    is_relevant: str
    iteration_count: int


# Global objects that are safe to reuse across calls
qdrant_client: QdrantClient = get_qdrant_client()
embedder: SentenceTransformer = get_embedder()
tavily_search = TavilySearch(api_key=TAVILY_API_KEY, topic="general", max_results=1)


def get_llm_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def get_llm_response(prompt: str) -> str:
    """
    Simple wrapper around the chat completion call.
    """
    client = get_llm_client()
    completion = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content or ""


def qdrant_search(
    client: QdrantClient,
    embedder: SentenceTransformer,
    collection_name: str,
    query: str,
    top_k: int = 3,
) -> str:
    """
    Run a semantic search on a collection and return joined texts
    based on the stored combined_text field in payload.
    """
    # Encode query to a vector
    query_vec = embedder.encode([query])[0].tolist()

    result = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        # You can add filters or search params later if needed
        # query_filter=models.Filter(must=[]),
        # search_params=models.SearchParams(hnsw_ef=128),
    )

    docs = []
    for scored_point in result.points:
        payload = scored_point.payload or {}
        text = payload.get("combined_text") or ""
        if text:
            docs.append(text)

    return "\n".join(docs)


#  Nodes for the graph


def retrieve_context_qna(state: GraphState) -> GraphState:
    print("--- RETRIEVING QNA CONTEXT ---")
    query = state["query"]
    context = qdrant_search(qdrant_client, embedder, QNA_COLLECTION, query, top_k=3)
    state["context"] = context
    state["source"] = "Retrieve_QnA"
    return state


def retrieve_context_device(state: GraphState) -> GraphState:
    print("--- RETRIEVING DEVICE CONTEXT ---")
    query = state["query"]
    context = qdrant_search(qdrant_client, embedder, DEVICE_COLLECTION, query, top_k=3)
    state["context"] = context
    state["source"] = "Retrieve_Device"
    return state


def tavily_web_search_node(state: GraphState) -> GraphState:
    print("--- WEB SEARCH CONTEXT ---")
    query = state["query"]

    try:
        result = tavily_search.invoke({"query": query})
        content = result["results"][0]["content"] if result.get("results") else ""
    except Exception as exc:
        # If Tavily fails we still want the pipeline to continue
        content = f"Tavily search failed with error {exc}. No external context available."

    state["context"] = content
    state["source"] = "Web_Search"
    return state


def router(state: GraphState) -> GraphState:
    """
    Decide which retriever to use based on the query.
    """
    query = state["query"]

    routing_prompt = f"""
    You are a routing assistant.

    Decide which source is most appropriate to answer the user question.

    Options
      Retrieve_QnA   general medical questions, symptoms, diagnosis, definitions, treatment
      Retrieve_Device   questions about medical devices, device models, indications for use, manufacturers
      Web_Search   questions about current events, very recent information, or anything clearly outside the Kaggle datasets

    User question
    {query}

    Reply with exactly one of
    Retrieve_QnA
    Retrieve_Device
    Web_Search
    """

    decision = get_llm_response(routing_prompt).strip()
    print(f"Router decision raw {decision}")

    # Small safety net
    if "Device" in decision:
        decision = "Retrieve_Device"
    elif "Web" in decision:
        decision = "Web_Search"
    elif "QnA" in decision or "QNA" in decision or "Qna" in decision:
        decision = "Retrieve_QnA"
    else:
        # Default to QnA when unsure
        decision = "Retrieve_QnA"

    state["source"] = decision
    return state


def route_decision(state: GraphState) -> str:
    return state["source"]


def check_context_relevance(state: GraphState) -> GraphState:
    """
    Check if retrieved context is relevant enough to the query.
    If not, the graph will send us to a web search as fallback.
    """
    print("--- CONTEXT RELEVANCE CHECKER ---")
    query = state["query"]
    context = state.get("context", "")

    relevance_prompt = f"""
    You check if the provided context is relevant to the user question.

    Context
    {context}

    Question
    {query}

    If the context has clear useful information for the question, answer with
    Yes

    If the context is off topic or unhelpful, answer with
    No
    """

    answer = get_llm_response(relevance_prompt).strip()
    answer = answer.split()[0] if answer else "Yes"
    answer = "Yes" if answer.lower().startswith("y") else "No"

    iteration_count = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration_count

    # Avoid endless loops
    if iteration_count >= 3:
        print("Max relevance iterations reached, forcing Yes")
        answer = "Yes"

    state["is_relevant"] = answer
    return state


def relevance_decision(state: GraphState) -> str:
    return state["is_relevant"]


def build_prompt(state: GraphState) -> GraphState:
    print("--- BUILDING PROMPT ---")
    query = state["query"]
    context = state.get("context", "")

    prompt = f"""
    You are a careful medical assistant.

    Use the context given below to answer the question.
    If the context is not enough, say that you are not sure and suggest that the user consult a medical professional.

    Context
    {context}

    Question
    {query}

    Reply briefly, around fifty words, and do not invent facts that are not supported by the context.
    """

    state["prompt"] = prompt
    return state


def call_llm_node(state: GraphState) -> GraphState:
    print("--- CALLING LLM ---")
    prompt = state["prompt"]
    answer = get_llm_response(prompt)
    state["response"] = answer
    return state


def build_workflow():
    """
    Build and compile the LangGraph workflow for the agentic RAG flow.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("Router", router)
    workflow.add_node("Retrieve_QnA", retrieve_context_qna)
    workflow.add_node("Retrieve_Device", retrieve_context_device)
    workflow.add_node("Web_Search", tavily_web_search_node)
    workflow.add_node("Relevance_Checker", check_context_relevance)
    workflow.add_node("Augment", build_prompt)
    workflow.add_node("Generate", call_llm_node)

    workflow.add_edge(START, "Router")

    workflow.add_conditional_edges(
        "Router",
        route_decision,
        {
            "Retrieve_QnA": "Retrieve_QnA",
            "Retrieve_Device": "Retrieve_Device",
            "Web_Search": "Web_Search",
        },
    )

    workflow.add_edge("Retrieve_QnA", "Relevance_Checker")
    workflow.add_edge("Retrieve_Device", "Relevance_Checker")
    workflow.add_edge("Web_Search", "Relevance_Checker")

    workflow.add_conditional_edges(
        "Relevance_Checker",
        relevance_decision,
        {
            "Yes": "Augment",
            "No": "Web_Search",
        },
    )

    workflow.add_edge("Augment", "Generate")
    workflow.add_edge("Generate", END)

    return workflow.compile()


agentic_rag = build_workflow()


def answer_query(query: str) -> str:
    """
    Public entry point that Chainlit uses.
    """
    initial_state: GraphState = {
        "query": query,
        "context": "",
        "prompt": "",
        "response": "",
        "source": "",
        "is_relevant": "No",
        "iteration_count": 0,
    }

    last_step_state: Dict[str, Any] | None = None

    # LangGraph stream yields steps in order
    for step in agentic_rag.stream(initial_state):
        last_step_state = step

    if not last_step_state:
        return "No response was generated, something went wrong."

    # last_step_state is a dict mapping node name to state
    node_state = list(last_step_state.values())[0]
    return node_state.get("response", "No response found.")
'''

from typing import TypedDict, Dict, Any

import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch

from qdrant_utils import get_qdrant_client, get_embedder

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

QNA_COLLECTION = "medical_qna"
DEVICE_COLLECTION = "medical_device_manual"


class GraphState(TypedDict):
    query: str
    context: str
    prompt: str
    response: str
    source: str
    iteration_count: int


# global objects reused across calls

qdrant_client: QdrantClient = get_qdrant_client()
embedder: SentenceTransformer = get_embedder()
tavily_search = TavilySearch(api_key=TAVILY_API_KEY, topic="general", max_results=1)


def get_llm_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def get_llm_response(prompt: str) -> str:
    """
    Simple wrapper for chat completion.
    """
    client = get_llm_client()
    completion = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content or ""


def qdrant_search_single(
    client: QdrantClient,
    embedder: SentenceTransformer,
    collection_name: str,
    query: str,
    top_k: int = 3,
) -> str:
    """
    Run a semantic search on a single collection and return
    a joined text based on the combined_text field.
    """
    query_vec = embedder.encode([query])[0].tolist()

    result = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )

    docs = []
    for scored_point in result.points:
        payload = scored_point.payload or {}
        text = payload.get("combined_text") or ""
        if text:
            docs.append(text)

    return "\n".join(docs)


def retrieve_context_from_vectors(state: GraphState) -> GraphState:
    """
    First step, always try to answer from vector store.

    We query both collections and concatenate the contexts.
    """
    print("--- RETRIEVING VECTOR CONTEXT ---")
    query = state["query"]

    qna_context = qdrant_search_single(
        qdrant_client, embedder, QNA_COLLECTION, query, top_k=3
    )
    device_context = qdrant_search_single(
        qdrant_client, embedder, DEVICE_COLLECTION, query, top_k=3
    )

    combined_context_parts = []
    if qna_context:
        combined_context_parts.append("QnA data:\n" + qna_context)
    if device_context:
        combined_context_parts.append("Device data:\n" + device_context)

    combined_context = "\n\n".join(combined_context_parts)

    state["context"] = combined_context
    state["source"] = "Vector_DB"
    return state


def use_web_search_if_needed(state: GraphState) -> GraphState:
    """
    If vector context is empty or extremely short, we switch to web search.
    Otherwise we keep vector context as is.
    """
    print("--- CHECKING IF WEB FALLBACK IS NEEDED ---")

    context = state.get("context", "") or ""
    query = state["query"]

    # very small or empty context, just go to web
    if len(context.strip()) < 40:
        print("Context too small, using web search.")
        try:
            result = tavily_search.invoke({"query": query})
            content = result["results"][0]["content"] if result.get("results") else ""
        except Exception as exc:
            content = f"Tavily search failed with error {exc}. No external context available."

        state["context"] = content
        state["source"] = "Web_Search"
        return state

    # context seems non empty, keep vector result
    print("Vector context seems acceptable, staying with vector store.")
    return state


def build_prompt(state: GraphState) -> GraphState:
    print("--- BUILDING PROMPT ---")
    query = state["query"]
    context = state.get("context", "")

    prompt = f"""
    You are a careful medical assistant.

    Use the context given below to answer the question.
    If the context is not enough, say that you are not sure and suggest that the user consult a medical professional.

    Context
    {context}

    Question
    {query}

    Reply briefly, around fifty words, and do not invent facts that are not supported by the context.
    """

    state["prompt"] = prompt
    return state


def call_llm_node(state: GraphState) -> GraphState:
    print("--- CALLING LLM ---")
    prompt = state["prompt"]
    answer = get_llm_response(prompt)
    state["response"] = answer
    return state


def build_workflow():
    """
    Build and compile the LangGraph workflow.

    Flow:
      START -> Retrieve_Vector -> Maybe_Web_Fallback -> Build_Prompt -> Generate -> END
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("Retrieve_Vector", retrieve_context_from_vectors)
    workflow.add_node("Maybe_Web_Fallback", use_web_search_if_needed)
    workflow.add_node("Build_Prompt", build_prompt)
    workflow.add_node("Generate", call_llm_node)

    workflow.add_edge(START, "Retrieve_Vector")
    workflow.add_edge("Retrieve_Vector", "Maybe_Web_Fallback")
    workflow.add_edge("Maybe_Web_Fallback", "Build_Prompt")
    workflow.add_edge("Build_Prompt", "Generate")
    workflow.add_edge("Generate", END)

    return workflow.compile()


agentic_rag = build_workflow()


def answer_query(query: str) -> str:
    """
    Public entry point that Chainlit uses.
    """
    initial_state: GraphState = {
        "query": query,
        "context": "",
        "prompt": "",
        "response": "",
        "source": "",
        "iteration_count": 0,
    }

    last_step_state: Dict[str, Any] | None = None

    for step in agentic_rag.stream(initial_state):
        last_step_state = step

    if not last_step_state:
        return "No response was generated, something went wrong."

    node_state = list(last_step_state.values())[0]
    return node_state.get("response", "No response found.")
