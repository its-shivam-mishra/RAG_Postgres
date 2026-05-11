"""
search.py
---------
Tavily web-search fallback used when the RAG pipeline cannot answer
from the user's own documents.

Strategy
--------
1. Call Tavily with ``include_answer=True`` — it tries to synthesise a
   direct, clean answer.  If present, return it immediately.
2. If Tavily's ``answer`` field is empty (common for Wikipedia-heavy
   queries), clean the raw snippets (strip markup / citation noise) and
   pass them through the LLM to produce readable prose.
"""

import re
import logging
from langchain_core.prompts import ChatPromptTemplate

from app.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS
from app.models import llm

logger = logging.getLogger(__name__)

# Phrases that signal the LLM could not answer from RAG context
UNCERTAINTY_PHRASES: list[str] = [
    "i don't know",
    "i do not know",
    "not mentioned",
    "not provided",
    "no information",
    "cannot find",
    "can't find",
    "not found in",
    "not in the context",
    "not available in",
    "the context does not",
    "the provided context",
    "based on the provided",
    "no relevant information",
]


def is_uncertain_answer(answer: str) -> bool:
    """Return True if the LLM signalled it could not answer from context."""
    lower = answer.lower()
    return any(phrase in lower for phrase in UNCERTAINTY_PHRASES)


def _clean_snippet(text: str, max_chars: int = 600) -> str:
    """Strip Wikipedia-style markup noise from a raw snippet."""
    text = re.sub(r'\^\s*_[a-z_]+_', '', text)   # ^_a__b_ citation anchors
    text = re.sub(r'\[\s*\.\.\.\s*\]', '', text)  # [...] truncation markers
    text = re.sub(r'\[\d+\]', '', text)            # [1] reference numbers
    text = re.sub(r'#{1,6}\s*', '', text)          # ## markdown headings
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]


def _summarise_with_llm(question: str, snippets: list[str]) -> str:
    """Use the LLM to turn cleaned web snippets into a concise answer."""
    context = "\n\n".join(
        f"[Source {i + 1}] {snip}" for i, snip in enumerate(snippets)
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Using only the web search excerpts "
            "below, answer the user's question in clear, concise prose "
            "(3-5 sentences max). Do NOT include citation numbers, markdown "
            "headings, or references. If the excerpts do not contain enough "
            "information, say so briefly.\n\n{context}",
        ),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return chain.invoke({"input": question, "context": context}).content.strip()


def tavily_search(question: str) -> tuple[str, list[str]]:
    """
    Search the web via Tavily and return ``(answer_text, source_urls)``.

    Falls back to LLM summarisation if Tavily does not provide a direct
    answer, ensuring the user always receives clean prose instead of raw
    markup.
    """
    from tavily import TavilyClient

    if not TAVILY_API_KEY or TAVILY_API_KEY == "YOUR_TAVILY_API_KEY":
        logger.error("[Search] TAVILY_API_KEY is not configured.")
        return (
            "Web search is not available: TAVILY_API_KEY is not configured.",
            [],
        )

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query=question,
        search_depth="advanced",
        max_results=TAVILY_MAX_RESULTS,
        include_answer=True,
    )

    results = response.get("results", [])
    sources: list[str] = [r["url"] for r in results if r.get("url")]

    if not results:
        return "No results found via web search.", []

    # ── Prefer Tavily's synthesised answer (already clean prose) ─────────────
    tavily_answer = (response.get("answer") or "").strip()
    if tavily_answer:
        logger.info("[Search] Returning Tavily direct answer.")
        return tavily_answer, sources

    # ── Fallback: clean snippets + LLM summarisation ─────────────────────────
    logger.info("[Search] No direct answer; summarising snippets via LLM.")
    cleaned = [
        _clean_snippet(r.get("content", ""))
        for r in results[:4]
        if r.get("content", "").strip()
    ]
    if not cleaned:
        return "No usable content found in web search results.", sources

    answer = _summarise_with_llm(question, cleaned)
    return answer, sources
