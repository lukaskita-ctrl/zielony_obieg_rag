# 🌿 Zielony Obieg RAG — AI Document Assistant for Waste Management

A fully local RAG (Retrieval-Augmented Generation) system built for a real waste management company in Poland. The assistant answers questions about contracts, lab reports, regulations, and operational documents — entirely offline, with zero data leaving the machine.

## 🎯 Why This Exists

Running a sewage sludge management company means dealing with dozens of legal documents, lab reports, land use agreements, environmental regulations, and waste transfer records. Finding the right information across 60+ files in the middle of a field or during a tender deadline is painful.

This RAG system lets me ask natural language questions in Polish and get accurate answers sourced directly from my company documents.

## ✨ Features

- **Polish-language RAG** on real legal/technical documents (contracts, lab reports, environmental permits)
- **Fine-tuned Gemma 2 9B** trained on domain-specific waste management data (93 examples)
- **BGE-M3 embeddings** — state-of-the-art multilingual model, excellent for Polish legal text
- **100% local** — no API calls, no cloud, no data leaves the machine
- **Gradio UI** — simple chat interface accessible via browser
- **Multi-format support** — ingests PDF, DOCX, DOC files
- **Incremental updates** — add new documents without rebuilding the entire vector database
- **Query expansion** — LLM reformulates questions for better retrieval (optional)

## 🏗️ Architecture

```
User Question (Polish)
        │
        ▼
   ┌─────────┐
   │ BGE-M3  │  → converts question to 1024-dim vector
   └────┬────┘
        │
        ▼
   ┌──────────┐
   │ ChromaDB │  → finds top-k most relevant document chunks
   └────┬─────┘
        │
        ▼
   ┌──────────────────────┐
   │ Fine-tuned Gemma 2 9B│  → generates answer from retrieved context
   │ (LoRA adapter)       │
   └──────────┬───────────┘
              │
              ▼
        Answer (Polish)
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemma 2 9B (fine-tuned with Unsloth + LoRA) |
| Embeddings | BAAI/bge-m3 via sentence-transformers |
| Vector Store | ChromaDB |
| Document Loading | LangChain (PyPDFLoader, Docx2txtLoader) |
| Fine-tuning | Unsloth + QLoRA (4-bit, 0.58% params trained) |
| UI | Gradio |
| Runtime | Python 3.11, PyTorch, CUDA |

## 📁 Project Structure

```
zielony-obieg-rag/
├── app.py                  # Main RAG application + Gradio UI
├── aktualizuj_baze.py      # Incremental database update script
├── train.py                # Fine-tuning script (Unsloth + LoRA)
├── export.py               # Model export to GGUF
├── requirements.txt        # Python dependencies
├── .gitignore
├── screenshots/
│   └── demo.png
└── docs/                   # Place your documents here
    └── (not included — add your own PDF/DOCX files)
```

## 🚀 Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with 12+ GB VRAM
- CUDA 12.x
- Conda (recommended)

### Installation

```bash
git clone https://github.com/lukaskita-ctrl/zielony-obieg-rag.git
cd zielony-obieg-rag

conda create -n rag python=3.11 -y
conda activate rag

pip install -r requirements.txt
```

### Configuration

Edit paths in `app.py`:

```python
DOCS_PATH = r"path\to\your\documents"
CHROMA_PATH = r"path\to\chroma_db"
MODEL_PATH = r"path\to\fine_tuned_model"
```

### Building the Vector Database

Place your PDF/DOCX files in the documents folder, then run the initial build (uncomment the build section in `app.py`):

```python
docs = load_documents()
chunks = chunk_documents(docs)
vectorstore = create_vector_store(chunks)
```

### Adding New Documents

Drop new files into the documents folder and run:

```bash
python aktualizuj_baze.py
```

Only new files are processed — existing documents are skipped.

### Running the Assistant

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

## 🎓 Fine-tuning

The model was fine-tuned on 93 domain-specific examples covering:

- Waste management regulations (R10 recovery process)
- Sewage sludge application procedures
- BDO (Polish Waste Database) operations
- Land use agreements and consortium contracts
- Laboratory analysis interpretation
- Dosage calculations and compliance

Training config: QLoRA 4-bit, rank 16, 3 epochs, batch size 4, learning rate 2e-4.

```bash
python train.py
```

## 📊 Results

| Query | Base Gemma 2 9B | Fine-tuned Gemma 2 9B |
|-------|----------------|----------------------|
| "Kim jesteś?" | Generic response | "Jestem asystentem firmy ZIELONY OBIEG..." |
| "Znajdź umowę konsorcjum" | "Nie wiem" | Correct contract details + parties |
| "Kto jest w konsorcjum?" | "Nie wiem" | All 4 members with roles |
| Polish legal terminology | Often confused | Domain-accurate responses |

## 🔑 Key Learnings

- **Embedding model matters more than LLM** for retrieval quality — switching from nomic-embed-text to BGE-M3 was the biggest improvement
- **Chunk size affects everything** — 1500 chars with 150 overlap works well for Polish legal documents
- **Fine-tuning on 93 examples** made a visible difference in domain understanding and response style
- **Polish NLP is hard** — most embedding models are English-first; multilingual models (BGE-M3) are essential
- **Local-first is viable** — 12 GB VRAM runs fine-tuned 9B model + embeddings without cloud dependencies

## 🗺️ Roadmap

- [ ] Voice interface (Whisper STT + edge-tts)
- [ ] Integration with BDO Monitor (waste transfer card management)
- [ ] Land parcel capacity calculator
- [ ] Hybrid search (BM25 + vector)
- [ ] Automated monthly reporting

## 🔒 Privacy

This project processes sensitive business documents (contracts, personal data, government permits). The entire pipeline runs locally — no data is sent to external APIs or cloud services.

## 👤 About

Built by **Łukasz Kita** ([@lukaskita-ctrl](https://github.com/lukaskita-ctrl)) — founder of Zielony Obieg, a waste management company in Poland specializing in sewage sludge transport and agricultural R10 recovery.

This project combines a real business need with an IBM Coursera course on Generative AI with Python, proving that the best way to learn AI is to solve your own problems.

## 📄 License

MIT

