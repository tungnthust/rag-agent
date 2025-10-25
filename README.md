# RAG Agent - Modular RAG System

A production-ready, modular Retrieval-Augmented Generation (RAG) system designed for easy maintenance and improvement. Built with modern tools and best practices for NLP and LLM applications.

## 🌟 Features

- **Modular Architecture**: Separate modules for ingestion, retrieval, and LLM serving
- **One-Time Ingestion**: Process and index documents once, query many times
- **Hybrid Search**: Combines BM25 (keyword-based) and vector search for robust retrieval
- **Advanced Chunking**: Parent-child chunking strategy for better context
- **Reranking**: Cross-encoder reranking for improved relevance
- **Optimized LLM Serving**: vLLM integration for high-throughput inference
- **Query Enhancement**: HyDE and query decomposition support
- **Structured Outputs**: JSON responses with citations and confidence scores
- **Persistent Storage**: ChromaDB for vector storage

## 📋 Architecture

```
rag-agent/
├── src/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   └── utils.py           # Common utilities
│   ├── modules/
│   │   ├── ingest.py          # Document ingestion (run once)
│   │   ├── retrieve.py        # Hybrid retrieval + reranking
│   │   ├── llm_server.py      # LLM serving (vLLM/HuggingFace)
│   │   └── rag_pipeline.py    # Complete RAG orchestration
├── main.py                    # CLI entry point
├── config_rag.py              # Legacy config (for compatibility)
├── question_answering.py      # Legacy monolithic code
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tungnthust/rag-agent.git
cd rag-agent

# Install dependencies
pip install -r requirements.txt

# Optional: Install vLLM for optimized serving (recommended)
pip install vllm
```

### Usage

#### 1. Ingest Documents (One-Time Setup)

Process and index your markdown documents:

```bash
python main.py ingest --docs-dir /path/to/documents
```

Options:
- `--docs-dir`: Directory containing markdown files
- `--force-reindex`: Force rebuild of index
- `--preset`: Use preset config (fast/balanced/accurate)

#### 2. Query the System

Ask a single question:

```bash
python main.py query --question "What is a resistor in electronics?"
```

Options:
- `--question`: Your question
- `--no-hyde`: Disable HyDE query enhancement
- `--decompose`: Enable query decomposition
- `--preset`: Use preset config

#### 3. Process Multiple Questions

Process questions from a CSV file:

```bash
python main.py process --questions questions.csv --output answers.csv
```

CSV format:
```csv
Question,A,B,C,D
"What is a resistor?",Component that resists current,A battery,A wire,None
```

## 🔧 Configuration

### Presets

Three built-in presets for different use cases:

- **fast**: Quick inference with smaller models (phi-2, MiniLM)
- **balanced**: Good quality with reasonable speed (Qwen-3B, BGE-small) [Default]
- **accurate**: Best quality with larger models (Mistral-7B, BGE-large)

### Custom Configuration

Create a custom config:

```python
from src import RAGConfig, LLMConfig, EmbeddingConfig

config = RAGConfig(
    llm=LLMConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        use_vllm=True,  # Use vLLM for serving
        temperature=0.1,
    ),
    embedding=EmbeddingConfig(
        model_name="BAAI/bge-small-en-v1.5",
    ),
    use_hyde=True,  # Enable HyDE
)
```

### Environment Variables

Configure via environment variables (prefix with `RAG_`):

```bash
export RAG_LLM_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
export RAG_LLM_USE_VLLM=true
export RAG_EMBEDDING_MODEL_NAME="BAAI/bge-large-en-v1.5"
```

## 📚 Programmatic Usage

```python
from src import RAGPipeline, load_config

# Load configuration
config = load_config(preset="balanced")

# Initialize pipeline
pipeline = RAGPipeline(config)

# One-time: Ingest documents
pipeline.ingest_documents()

# Or: Load existing index
pipeline.load_index()

# Query
answer = pipeline.query("What is a resistor?")

print(answer.answer)
for citation in answer.citations:
    print(f"- {citation.document_name}: {citation.section_header}")
```

## 🧩 Module Details

### 1. Ingestion Module (`src/modules/ingest.py`)

- **Purpose**: Process and index documents (run once)
- **Features**:
  - Markdown parsing with header metadata
  - Parent-child chunking strategy
  - ChromaDB vector storage
  - Persistent indexing

**Usage**:
```python
from src.modules import DocumentIngestion

ingestion = DocumentIngestion(config)
parent_nodes, child_nodes = ingestion.ingest_documents()
```

### 2. Retrieval Module (`src/modules/retrieve.py`)

- **Purpose**: Hybrid search and reranking
- **Features**:
  - BM25 keyword search
  - Dense vector search
  - Cross-encoder reranking
  - Small-to-big context retrieval

**Usage**:
```python
from src.modules import HybridRetriever

retriever = HybridRetriever(config)
context_nodes, results = retriever.retrieve("What is a resistor?")
```

### 3. LLM Server Module (`src/modules/llm_server.py`)

- **Purpose**: Optimized LLM inference
- **Features**:
  - vLLM integration for high throughput
  - Fallback to HuggingFace Transformers
  - Query preprocessing (HyDE, decomposition)
  - Structured answer generation

**Usage**:
```python
from src.modules import LLMServer

llm_server = LLMServer(config)
response = llm_server.generate("Explain resistors")
```

## 🔬 Advanced Features

### HyDE (Hypothetical Document Embeddings)

Improve retrieval by generating hypothetical answers:

```python
answer = pipeline.query(question, use_hyde=True)
```

### Query Decomposition

Break complex queries into sub-queries:

```python
answer = pipeline.query(question, use_decomposition=True)
```

### Parent-Child Chunking

Retrieve at paragraph level, provide section-level context:

```python
# Automatically handles small-to-big context retrieval
context_nodes = retriever.retrieve_parent_chunks(child_nodes)
```

## 🎯 Performance Optimization

### vLLM Setup

vLLM provides significantly faster inference:

```python
config = RAGConfig(
    llm=LLMConfig(
        use_vllm=True,
        tensor_parallel_size=1,  # Multi-GPU support
        gpu_memory_utilization=0.9,
    )
)
```

### Quantization

Reduce memory usage with quantization:

```python
config = RAGConfig(
    llm=LLMConfig(
        quantization="awq",  # or "gptq"
    )
)
```

## 🧪 Testing

Run individual modules:

```bash
# Test ingestion
python -m src.modules.ingest

# Test retrieval
python -m src.modules.retrieve

# Test LLM server
python -m src.modules.llm_server
```

## 📊 Comparison with Legacy Code

| Aspect | Legacy | Modular |
|--------|--------|---------|
| Structure | Monolithic | Modular |
| Ingestion | Mixed with query | Separate module |
| LLM Serving | Basic HF | vLLM optimized |
| Vector Store | In-memory | Persistent (Chroma) |
| Config | Scattered | Centralized (Pydantic) |
| Testability | Difficult | Easy |
| Maintenance | Hard | Easy |

## 🔍 Troubleshooting

### vLLM Installation Issues

If vLLM installation fails, the system automatically falls back to HuggingFace Transformers:

```bash
# Try installing with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

Reduce memory usage:
- Use smaller models (preset="fast")
- Enable quantization
- Reduce batch sizes in config

### Index Not Found

Run ingestion before querying:

```bash
python main.py ingest --docs-dir /path/to/documents
```

## 🤝 Contributing

This modular architecture makes contributions easy:

1. Add new retrieval strategies in `src/modules/retrieve.py`
2. Add new LLM backends in `src/modules/llm_server.py`
3. Add new chunking strategies in `src/modules/ingest.py`

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

Built with modern, production-ready tools:
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [LlamaIndex](https://github.com/run-llama/llama_index) - RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - Embeddings
- [BM25](https://github.com/dorianbrown/rank_bm25) - Keyword search

## 📧 Contact

For questions or issues, please open a GitHub issue.
