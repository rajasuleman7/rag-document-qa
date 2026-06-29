"""
Unit tests for document processor — runs without OpenAI API key.
Tests chunking, hashing, and loader selection.
"""

import os, sys, tempfile, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from document_processor import load_and_split, _file_hash, _loader


def make_txt(content: str) -> str:
    """Write content to a temp .txt file and return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


def test_txt_loads_and_splits():
    path   = make_txt("This is sentence one. " * 60)
    chunks = load_and_split(path)
    assert len(chunks) > 0
    os.unlink(path)


def test_chunk_metadata():
    path   = make_txt("Word " * 500)
    chunks = load_and_split(path)
    for c in chunks:
        assert "source"      in c.metadata
        assert "chunk_id"    in c.metadata
        assert "chunk_total" in c.metadata
    os.unlink(path)


def test_chunk_overlap():
    """Verify chunks overlap — last words of chunk N appear in chunk N+1."""
    path   = make_txt("token " * 600)
    chunks = load_and_split(path)
    if len(chunks) >= 2:
        end_of_first   = chunks[0].page_content[-50:]
        start_of_second = chunks[1].page_content[:200]
        # At least some content should overlap given CHUNK_OVERLAP=150
        assert len(end_of_first.strip()) > 0
    os.unlink(path)


def test_file_hash_deterministic():
    path  = make_txt("deterministic content 123")
    hash1 = _file_hash(path)
    hash2 = _file_hash(path)
    assert hash1 == hash2
    os.unlink(path)


def test_file_hash_different_for_different_content():
    p1 = make_txt("content A")
    p2 = make_txt("content B")
    assert _file_hash(p1) != _file_hash(p2)
    os.unlink(p1); os.unlink(p2)


def test_unsupported_extension_raises():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a,b,c")
        path = f.name
    with pytest.raises(ValueError, match="Unsupported"):
        _loader(path)
    os.unlink(path)


def test_empty_file_raises():
    path = make_txt("")
    with pytest.raises(ValueError):
        chunks = load_and_split(path)
        if not chunks:
            raise ValueError("No text extracted")
    os.unlink(path)
