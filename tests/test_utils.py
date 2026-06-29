"""Unit tests for utility functions."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    count_tokens_approx, suggested_questions,
    export_chat, clean_text_preview,
)


def test_token_count_positive():
    assert count_tokens_approx("hello world") > 0


def test_token_count_scales():
    short = count_tokens_approx("hi")
    long  = count_tokens_approx("hello world " * 100)
    assert long > short


def test_suggested_questions_returns_list():
    qs = suggested_questions("This paper presents an abstract and methodology.")
    assert isinstance(qs, list)
    assert len(qs) >= 3


def test_suggested_questions_research_trigger():
    qs = suggested_questions("abstract introduction methodology results conclusion")
    texts = " ".join(qs).lower()
    assert "method" in texts or "result" in texts or "finding" in texts


def test_suggested_questions_legal_trigger():
    qs = suggested_questions("This agreement between the parties herein sets forth terms and clauses.")
    texts = " ".join(qs).lower()
    assert "terminat" in texts or "obligat" in texts or "liabilit" in texts


def test_export_chat_contains_content():
    msgs = [
        {"role": "user",      "content": "What is this?", "sources": []},
        {"role": "assistant", "content": "It is a test.", "sources": [
            {"source": "doc.pdf", "page": 1, "excerpt": "some text…"}
        ]},
    ]
    md = export_chat(msgs, "doc.pdf")
    assert "What is this?" in md
    assert "It is a test." in md
    assert "doc.pdf" in md


def test_clean_text_preview_truncates():
    long_text = "word " * 200
    result    = clean_text_preview(long_text, max_chars=100)
    assert len(result) <= 104   # 100 + "…"
    assert result.endswith("…")


def test_clean_text_preview_cleans_whitespace():
    messy  = "lots   of\n\n\nwhitespace   here"
    result = clean_text_preview(messy)
    assert "  " not in result
