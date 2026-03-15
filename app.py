import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True

import hashlib
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from unsloth import FastLanguageModel
import gradio as gr

DOCS_PATH = r"C:\Users\lukas\rag_doc"
CHROMA_PATH = r"C:\Users\lukas\rag_projekt\chroma_db"
MODEL_PATH = r"C:\Users\lukas\fine_tuning\zielony_obieg_gemma"

# załaduj LLM raz przy starcie
print("Ładuję fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model załadowany!")

def generate(prompt):
    system_msg = "Jesteś asystentem firmy ZIELONY OBIEG ŁUKASZ KITA, specjalizującej się w zagospodarowaniu komunalnych osadów ściekowych na cele rolnicze w województwie mazowieckim."
    
    messages = [
        {"role": "user", "content": system_msg + "\n\n" + prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=1900
).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.2,
    )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response

def load_documents():
    docs = []
    for filename in os.listdir(DOCS_PATH):
        filepath = os.path.join(DOCS_PATH, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs.extend(loader.load())
            elif filename.lower().endswith((".docx", ".doc")):
                loader = Docx2txtLoader(filepath)
                docs.extend(loader.load())
        except Exception as e:
            print(f"Błąd przy {filename}: {e}")
    print(f"Załadowano {len(docs)} fragmentów z {DOCS_PATH}")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Podzielono na {len(chunks)} chunków")
    return chunks

def chat(question, history, vectorstore):
    results = vectorstore.similarity_search(question, k=3)

    # debug
    print(f"\n--- Pytanie: {question} ---")
    for i, doc in enumerate(results):
        src = doc.metadata.get('source', '?').split('\\')[-1]
        print(f"  {i+1}. {src}: {doc.page_content[:100]}")

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Na podstawie poniższych dokumentów odpowiedz na pytanie po polsku.
Jeśli nie znasz odpowiedzi na podstawie dokumentów, powiedz że nie wiesz.

Dokumenty:
{context}

Pytanie: {question}

Odpowiedź:"""

    return generate(prompt)

if __name__ == "__main__":
    print("\n--- ETAP 1: INICJALIZACJA BAZY WIEDZY ---")

    print("Ładuję model embeddingów (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print(f"Znaleziono istniejącą bazę w {CHROMA_PATH}. Wczytywanie...")
        active_vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        print(f"Baza wczytana. Liczba dokumentów: {active_vectorstore._collection.count()}")
    else:
        print("Baza nie istnieje. Buduję nową...")
        docs = load_documents()
        chunks = chunk_documents(docs)

        # deduplikacja MD5
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            chunk_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                unique_chunks.append(chunk)
        print(f"Po deduplikacji: {len(unique_chunks)} unikalnych chunków (było {len(chunks)})")

        active_vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        for i, chunk in enumerate(unique_chunks):
            try:
                active_vectorstore.add_documents([chunk])
            except Exception as e:
                print(f"Pominięto chunk {i}: {e}")

        print("Nowa baza gotowa!")

    print("\n--- ETAP 2: URUCHAMIANIE INTERFEJSU ---")

    def gradio_chat_wrapper(question, history):
        return chat(question, history, active_vectorstore)

    demo = gr.ChatInterface(
        fn=gradio_chat_wrapper,
        title="Zielony Obieg RAG",
        description="Asystent dokumentów firmy Zielony Obieg — fine-tuned Gemma 2 9B + BAAI/bge-m3",
    )
    demo.launch()
