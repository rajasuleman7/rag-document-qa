"""
Configuration — model choices, chunking params, UI settings.
"""

SUPPORTED_MODELS = {
    "gpt-3.5-turbo":  "GPT-3.5 Turbo  (fast, cheap)",
    "gpt-4o-mini":    "GPT-4o Mini    (smart, cheap)",
    "gpt-4o":         "GPT-4o         (best, higher cost)",
}

DEFAULT_MODEL   = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K_CHUNKS  = 5

MAX_FILE_SIZE_MB = 20
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

APP_TITLE   = "DocMind — RAG Document Q&A"
APP_ICON    = "🧠"
APP_TAGLINE = "Upload any document. Ask anything. Get grounded answers with citations."
