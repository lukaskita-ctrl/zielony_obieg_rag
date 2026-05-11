import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

torch._dynamo.config.disable = True

import gradio as gr
from unsloth import FastLanguageModel

from rag_utils import (
    CHROMA_PATH,
    DOCS_PATH,
    MODEL_PATH,
    add_chunks_to_store,
    chunk_documents,
    dedupe_chunks,
    get_embeddings,
    get_vectorstore,
    load_documents,
)


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
    system_msg = (
        "Jesteś asystentem firmy ZIELONY OBIEG ŁUKASZ KITA, specjalizującej się "
        "w zagospodarowaniu komunalnych osadów ściekowych na cele rolnicze "
        "w województwie mazowieckim."
    )

    messages = [{"role": "user", "content": system_msg + "\n\n" + prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1900
    ).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.2,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def chat(question, history, vectorstore):
    results = vectorstore.similarity_search(question, k=3)

    sources = [os.path.basename(doc.metadata.get("source", "?")) for doc in results]

    print(f"\n--- Pytanie: {question} ---")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {sources[i]}: {doc.page_content[:100]}")

    context = "\n\n".join(doc.page_content for doc in results)

    prompt = f"""Na podstawie poniższych dokumentów odpowiedz na pytanie po polsku.
Jeśli nie znasz odpowiedzi na podstawie dokumentów, powiedz że nie wiesz.

Dokumenty:
{context}

Pytanie: {question}

Odpowiedź:"""

    answer = generate(prompt)

    unique_sources = sorted({s for s in sources if s and s != "?"})
    if unique_sources:
        answer += "\n\n**Źródła:**\n" + "\n".join(f"- {s}" for s in unique_sources)
    return answer


def build_vectorstore_from_scratch(embeddings):
    print("Baza nie istnieje. Buduję nową...")
    docs, _ = load_documents(DOCS_PATH)
    print(f"Załadowano {len(docs)} fragmentów z {DOCS_PATH}")
    chunks = chunk_documents(docs)
    print(f"Podzielono na {len(chunks)} chunków")
    unique = dedupe_chunks(chunks)
    print(f"Po deduplikacji: {len(unique)} unikalnych chunków (było {len(chunks)})")

    vectorstore = get_vectorstore(embeddings)
    add_chunks_to_store(vectorstore, unique)
    print("Nowa baza gotowa!")
    return vectorstore


if __name__ == "__main__":
    print("\n--- ETAP 1: INICJALIZACJA BAZY WIEDZY ---")

    print("Ładuję model embeddingów (BAAI/bge-m3)...")
    embeddings = get_embeddings()

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print(f"Znaleziono istniejącą bazę w {CHROMA_PATH}. Wczytywanie...")
        active_vectorstore = get_vectorstore(embeddings)
        print(
            f"Baza wczytana. Liczba dokumentów: "
            f"{active_vectorstore._collection.count()}"
        )
    else:
        active_vectorstore = build_vectorstore_from_scratch(embeddings)

    print("\n--- ETAP 2: URUCHAMIANIE INTERFEJSU ---")

    def gradio_chat_wrapper(question, history):
        return chat(question, history, active_vectorstore)

    demo = gr.ChatInterface(
        fn=gradio_chat_wrapper,
        title="Zielony Obieg RAG",
        description="Asystent dokumentów firmy Zielony Obieg — fine-tuned Gemma 2 9B + BAAI/bge-m3",
    )
    demo.launch()
