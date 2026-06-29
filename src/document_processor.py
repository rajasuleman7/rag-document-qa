"""
Document Processor
PDF/TXT ingestion → chunking → OpenAI embeddings → FAISS vectorstore.
Caches vectorstores by file hash so the same document isn't re-embedded.
"""

import os, hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150


def _file_hash(fp: str) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _loader(fp: str):
    ext = Path(fp).suffix.lower()
    if   ext == ".pdf": return PyPDFLoader(fp)
    elif ext == ".txt": return TextLoader(fp, encoding="utf-8")
    raise ValueError(f"Unsupported type: {ext}. Use .pdf or .txt")


def load_and_split(fp: str) -> list:
    docs = _loader(fp).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    fname  = Path(fp).name
    for i, c in enumerate(chunks):
        c.metadata.update({"source": fname, "chunk_id": i, "chunk_total": len(chunks)})
    return chunks


def build_vectorstore(fp: str, api_key: str, persist: bool = True):
    fhash      = _file_hash(fp)
    store_path = os.path.join(VECTORSTORE_DIR, fhash)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    if persist and os.path.exists(store_path):
        vs = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        return vs, {"source": Path(fp).name, "cached": True}

    chunks = load_and_split(fp)
    if not chunks:
        raise ValueError("No text extracted — check the file isn't scanned/image-only.")

    vs = FAISS.from_documents(chunks, embeddings)
    if persist:
        os.makedirs(store_path, exist_ok=True)
        vs.save_local(store_path)

    return vs, {"source": Path(fp).name, "cached": False,
                "chunks": len(chunks), "file_hash": fhash}
