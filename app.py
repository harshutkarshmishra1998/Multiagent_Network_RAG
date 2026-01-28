"""
=================================================
Multi-Agent Iterative Corrective RAG
=================================================

Agents:
1. Planner        → understands the task
2. Rewrite        → query enhancement + multi-query
3. Retriever      → MMR + multi-query retrieval
4. Reranker       → relevance reranking
5. Answer         → grounded answer generation
6. Critic         → decides PASS / RETRY

Framework: LangGraph
Vector DB: FAISS
Embeddings: OpenAIEmbeddings
LLM: ChatGroq (qwen/qwen3-32b)
"""

# =================================================
# 1. Imports & API setup
# =================================================

import api_keys

from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field


# =================================================
# 2. Vector Store Builder
# =================================================

def build_vectorstore(urls: list[str]) -> FAISS:
    """
    Loads documents, cleans them, chunks them,
    and builds a FAISS vector store.
    """

    docs = []

    for url in urls:
        loaded = WebBaseLoader(
            url,
            continue_on_failure=True
        ).load()

        for d in loaded:
            text = (d.page_content or "").strip()
            if text and "Redirecting" not in text:
                docs.append(d)

    if not docs:
        raise ValueError("No documents loaded for vectorstore.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(
        chunks,
        OpenAIEmbeddings()
    )


# =================================================
# 3. Knowledge Base
# =================================================

DOC_URLS = [
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
]

VECTORSTORE = build_vectorstore(DOC_URLS)


# =================================================
# 4. Graph State
# =================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    enhanced_queries: list[str]
    context: str
    answer: str
    retries: int
    critic_decision: str

# =================================================
# 5. LLM
# =================================================

LLM = ChatGroq(model="qwen/qwen3-32b")


# =================================================
# 6. Planner Agent
# =================================================

def planner_agent(state: AgentState):
    """
    Understands the task (light planning hook).
    """

    question = state["messages"][0].content

    prompt = PromptTemplate(
        template="""
    Break the following task into clear steps:

    Task:
    {question}

    Steps:
    """,
        input_variables=["question"],
    )

    steps = (prompt | LLM | StrOutputParser()).invoke(
        {"question": question}
    )

    plan = [s.strip("- ") for s in steps.split("\n") if s.strip()]

    return {"plan": plan}

# =================================================
# 7. Rewrite Agent (Query Enhancement + Multi-Query)
# =================================================

def rewrite_agent(state: AgentState):
    """
    Improves retrieval by:
    - Refining the query
    - Generating multiple semantic variants
    """

    question = state["messages"][0].content

    prompt = PromptTemplate(
        template="""
Rewrite the question to improve retrieval quality.

1. Make it more explicit and specific.
2. Generate 3 alternative semantic variants.

Original question:
{question}

Return in this format:

REFINED:
<refined query>

VARIANTS:
- <variant 1>
- <variant 2>
- <variant 3>
""",
        input_variables=["question"],
    )

    output = (prompt | LLM | StrOutputParser()).invoke(
        {"question": question}
    )

    refined = ""
    variants = []

    for line in output.splitlines():
        if line.startswith("-"):
            variants.append(line.replace("-", "").strip())
        elif line.strip() and not line.startswith(("REFINED", "VARIANTS")):
            refined = line.strip()

    queries = [refined] + variants

    return {
        "enhanced_queries": queries,
        "retries": state["retries"] + 1,
    }


# =================================================
# 8. Retriever Agent (MMR + Multi-Query)
# =================================================

def retriever_agent(state: AgentState):
    """
    Retrieves documents using:
    - Multi-query expansion
    - MMR for diversity
    """

    queries = state.get("enhanced_queries") or [
        state["messages"][0].content
    ]

    retriever = VECTORSTORE.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )

    docs = []
    for q in queries:
        if isinstance(q, str):
            docs.extend(retriever.invoke(q))

    # Deduplicate
    seen = set()
    unique_docs = []
    for d in docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    context = "\n\n".join(d.page_content for d in unique_docs)

    return {"context": context}


# =================================================
# 9. Reranker Agent
# =================================================

def rerank_agent(state: AgentState):
    """
    Reranks retrieved chunks using LLM relevance judgment.
    """

    question = state["messages"][0].content
    chunks = state["context"].split("\n\n")

    prompt = PromptTemplate(
        template="""
Select the most relevant chunks for answering the question.
Return only the top relevant content.

Question:
{question}

Chunks:
{chunks}
""",
        input_variables=["question", "chunks"],
    )

    ranked = (prompt | LLM | StrOutputParser()).invoke(
        {"question": question, "chunks": "\n\n".join(chunks)}
    )

    return {"context": ranked}


# =================================================
# 10. Answer Agent
# =================================================

def answer_agent(state: AgentState):
    """
    Generates a grounded answer using retrieved context.
    """

    prompt = PromptTemplate(
        template="""
Answer the question using ONLY the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    answer = (prompt | LLM | StrOutputParser()).invoke(
        {
            "context": state["context"],
            "question": state["messages"][0].content,
        }
    )

    return {"answer": answer}


# =================================================
# 11. Critic Agent (Iteration Control)
# =================================================

def critic_agent(state: AgentState):
    """
    Evaluates answer quality and writes decision into state.
    """

    class Verdict(BaseModel):
        decision: str = Field(description="PASS or FAIL")

    grader = LLM.with_structured_output(Verdict)

    prompt = PromptTemplate(
        template="""
Is the following answer correct, complete,
and grounded in the context?

Answer:
{answer}

Reply PASS or FAIL.
""",
        input_variables=["answer"],
    )

    result = (prompt | grader).invoke(
        {"answer": state["answer"]}
    )

    decision = "final" if result.decision == "PASS" else "retry" # type: ignore

    return {
        "critic_decision": decision
    }


# =================================================
# 12. Build LangGraph
# =================================================

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_agent)
workflow.add_node("rewrite", rewrite_agent)
workflow.add_node("retrieve", retriever_agent)
workflow.add_node("rerank", rerank_agent)
workflow.add_node("answer_agent", answer_agent)
workflow.add_node("critic", critic_agent)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "answer_agent")
workflow.add_edge("answer_agent", "critic")

def route_from_critic(state: AgentState) -> Literal["final", "retry"]:
    if state["critic_decision"] == "final":
        return "final"
    return "retry"


workflow.add_conditional_edges(
    "critic",
    route_from_critic,
    {
        "final": END,
        "retry": "rewrite",
    }
)

graph = workflow.compile()


# =================================================
# 13. Runner
# =================================================

def run(question: str):
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "enhanced_queries": [],
        "context": "",
        "answer": "",
        "retries": 0,
    })

    print("\n==============================")
    print("QUESTION:", question)
    print("==============================")
    print(result["answer"])


run("Explain LangGraph and how it differs from LangChain")
