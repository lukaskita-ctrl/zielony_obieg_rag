from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

DOCS_PATH = r"C:\Users\lukas\rag_doc"
CHROMA_PATH = r"C:\Users\lukas\rag_projekt\chroma_db"
PROCESSED_FILE = r"C:\Users\lukas\rag_projekt\processed_files.txt"

def get_processed_files():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()

def save_processed_files(files):
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(files))

def load_new_documents():
    processed = get_processed_files()
    new_docs = []
    new_files = []
    
    for filename in os.listdir(DOCS_PATH):
        if filename in processed:
            continue
        filepath = os.path.join(DOCS_PATH, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                new_docs.extend(loader.load())
                new_files.append(filename)
            elif filename.lower().endswith((".docx", ".doc")):
                loader = Docx2txtLoader(filepath)
                new_docs.extend(loader.load())
                new_files.append(filename)
        except Exception as e:
            print(f"Błąd przy {filename}: {e}")
            new_files.append(filename)  # zapisz żeby nie próbować ponownie
    
    return new_docs, new_files, processed

def main():
    new_docs, new_files, processed = load_new_documents()
    
    if not new_docs:
        print("Brak nowych dokumentów do dodania.")
        return
    
    print(f"Znaleziono {len(new_files)} nowych plików")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(new_docs)
    print(f"Podzielono na {len(chunks)} chunków")
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    for i, chunk in enumerate(chunks):
        try:
            vectorstore.add_documents([chunk])
        except Exception as e:
            print(f"Pominięto chunk {i}: {e}")
    
    # zapisz przetworzone pliki
    all_processed = processed | set(new_files)
    save_processed_files(all_processed)
    print(f"Dodano {len(chunks)} chunków. Łącznie przetworzonych plików: {len(all_processed)}")

if __name__ == "__main__":
    main()