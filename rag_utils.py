"""Common helpers for Zielony Obieg RAG: paths, document loading, chunking, vector store."""

import hashlib
import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


HOME = os.path.expanduser("~")

DOCS_PATH = os.environ.get("ZO_DOCS_PATH", os.path.join(HOME, "rag_doc"))
CHROMA_PATH = os.environ.get(
    "ZO_CHROMA_PATH", os.path.join(HOME, "rag_projekt", "chroma_db")
)
PROCESSED_FILE = os.environ.get(
    "ZO_PROCESSED_FILE",
    os.path.join(HOME, "rag_projekt", "processed_files.txt"),
)
MODEL_PATH = os.environ.get(
    "ZO_MODEL_PATH", os.path.join(HOME, "fine_tuning", "zielony_obieg_gemma")
)
EMBEDDING_MODEL = os.environ.get("ZO_EMBEDDING_MODEL", "BAAI/bge-m3")

CHUNK_SIZE = int(os.environ.get("ZO_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("ZO_CHUNK_OVERLAP", "150"))


def load_documents(docs_path=DOCS_PATH, skip=None):
    """Load PDF/DOCX files from docs_path. Returns (docs, processed_filenames)."""
    skip = skip or set()
    docs = []
    processed = []
    for filename in os.listdir(docs_path):
        if filename in skip:
            continue
        filepath = os.path.join(docs_path, filename)
        lower = filename.lower()
        try:
            if lower.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif lower.endswith((".docx", ".doc")):
                loader = Docx2txtLoader(filepath)
            else:
                continue
            docs.extend(loader.load())
            processed.append(filename)
        except Exception as e:
            print(f"Błąd przy {filename}: {e}")
            processed.append(filename)
    return docs, processed


def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)


def dedupe_chunks(chunks):
    seen = set()
    unique = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique


def get_embeddings(model_name=EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vectorstore(embeddings=None, chroma_path=CHROMA_PATH):
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(persist_directory=chroma_path, embedding_function=embeddings)


def get_processed_files(path=PROCESSED_FILE):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


def save_processed_files(files, path=PROCESSED_FILE):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(files))


def add_chunks_to_store(vectorstore, chunks):
    for i, chunk in enumerate(chunks):
        try:
            vectorstore.add_documents([chunk])
        except Exception as e:
            print(f"Pominięto chunk {i}: {e}")
