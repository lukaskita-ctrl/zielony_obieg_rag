"""Incremental update: ingest only new PDF/DOCX files into the Chroma vector store."""

from rag_utils import (
    DOCS_PATH,
    add_chunks_to_store,
    chunk_documents,
    dedupe_chunks,
    get_processed_files,
    get_vectorstore,
    load_documents,
    save_processed_files,
)


def main():
    processed = get_processed_files()
    new_docs, new_files = load_documents(DOCS_PATH, skip=processed)

    if not new_docs:
        print("Brak nowych dokumentów do dodania.")
        return

    print(f"Znaleziono {len(new_files)} nowych plików")

    chunks = chunk_documents(new_docs)
    unique = dedupe_chunks(chunks)
    print(f"Podzielono na {len(unique)} unikalnych chunków (było {len(chunks)})")

    vectorstore = get_vectorstore()
    add_chunks_to_store(vectorstore, unique)

    all_processed = processed | set(new_files)
    save_processed_files(all_processed)
    print(
        f"Dodano {len(unique)} chunków. Łącznie przetworzonych plików: {len(all_processed)}"
    )


if __name__ == "__main__":
    main()
