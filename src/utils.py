"""
Utility helpers — token counting, suggested questions, chat export.
"""

import json
import re
from datetime import datetime


def count_tokens_approx(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def suggested_questions(doc_text_sample: str) -> list:
    """
    Generate starter questions from the first ~1000 chars of a document.
    Purely heuristic — no LLM call needed.
    """
    sample = doc_text_sample[:1000].lower()
    questions = []

    # Generic questions that work for most documents
    questions += [
        "What is the main topic of this document?",
        "Summarise the key points in bullet points.",
        "What are the most important conclusions or findings?",
    ]

    # Domain-specific triggers
    if any(w in sample for w in ["abstract", "introduction", "methodology", "results"]):
        questions += [
            "What methodology was used in this study?",
            "What were the main results or findings?",
            "What limitations are mentioned?",
        ]
    if any(w in sample for w in ["agreement", "terms", "clause", "party", "herein"]):
        questions += [
            "What are the key obligations of each party?",
            "What termination conditions are described?",
            "Are there any penalty or liability clauses?",
        ]
    if any(w in sample for w in ["revenue", "profit", "financial", "quarter", "earnings"]):
        questions += [
            "What are the key financial figures mentioned?",
            "What revenue or profit numbers are reported?",
            "What are the financial risks identified?",
        ]
    if any(w in sample for w in ["chapter", "section", "table of contents"]):
        questions += [
            "What topics does each chapter cover?",
            "What examples or case studies are discussed?",
        ]

    return questions[:6]


def export_chat(messages: list, doc_name: str) -> str:
    """Export chat history as a formatted markdown string."""
    lines = [
        f"# Document Q&A Export",
        f"**Document:** {doc_name}",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]
    for msg in messages:
        role    = "**You**" if msg["role"] == "user" else "**Assistant**"
        content = msg["content"]
        lines.append(f"{role}\n\n{content}\n")
        if msg.get("sources"):
            lines.append("*Sources:*")
            for s in msg["sources"]:
                lines.append(f"- {s['source']} p.{s['page']} — _{s['excerpt']}_")
        lines.append("---\n")
    return "\n".join(lines)


def clean_text_preview(raw: str, max_chars: int = 500) -> str:
    """Clean and truncate raw extracted text for preview display."""
    text = re.sub(r"\s+", " ", raw).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text
