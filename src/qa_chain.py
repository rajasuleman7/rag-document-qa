"""
QA Chain — built with LangChain Expression Language (LCEL)
RAG pipeline: retriever → prompt → LLM → answer with source citations.
Uses the modern LCEL Runnable interface with conversational memory.
"""

from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


SYSTEM_PROMPT = """You are an expert document analyst. Answer questions accurately \
and concisely using ONLY the context extracted from the uploaded document.

Rules:
- Base every answer strictly on the provided context. Do not use outside knowledge.
- If the answer is not in the context, say clearly: "This information is not found in the document."
- When referencing specific facts, mention the page or section if available.
- Use bullet points for lists. Be concise but complete.
- For multi-part questions, address each part separately.

Context from document:
{context}"""


def _format_docs(docs: list) -> str:
    """Concatenate retrieved chunks into a single context string."""
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_qa_chain(vectorstore, api_key: str, model: str = "gpt-3.5-turbo", k: int = 5):
    """
    Build a RAG chain using LCEL with in-memory conversation history.
    Returns (chain, history_list) — history_list is mutated each call.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0.1,
        openai_api_key=api_key,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.6},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | _format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def format_sources(source_docs: list) -> list:
    """Deduplicate and format retrieved chunks for the citation panel."""
    seen, out = set(), []
    for doc in source_docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", ""))
        if key not in seen:
            seen.add(key)
            out.append({
                "source":  doc.metadata.get("source", "document"),
                "page":    doc.metadata.get("page", "—"),
                "excerpt": doc.page_content[:250].strip() + "…",
            })
    return out


def ask(chain, retriever, question: str, history: list) -> dict:
    """
    Run a question through the LCEL chain.
    history: list of (human_msg, ai_msg) tuples for conversational context.
    Returns dict with answer and sources.
    """
    # Build message history (last 6 exchanges)
    messages = []
    for human, ai in history[-6:]:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))

    # Get source docs separately for citations
    source_docs = retriever.invoke(question)

    answer = chain.invoke({
        "question": question,
        "history":  messages,
    })

    sources = format_sources(source_docs)
    return {"answer": answer, "sources": sources}
