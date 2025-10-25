# Migration Guide

This guide helps you migrate from the legacy monolithic code to the new modular architecture.

## Overview of Changes

### Old Structure (Legacy)
```
rag-agent/
├── question_answering.py  # All-in-one monolithic file
└── config_rag.py          # Configuration
```

### New Structure (Modular)
```
rag-agent/
├── main.py                # CLI entry point
├── src/
│   ├── core/
│   │   ├── config.py      # Pydantic-based configuration
│   │   └── utils.py       # Shared utilities
│   └── modules/
│       ├── ingest.py      # Document ingestion (run once)
│       ├── retrieve.py    # Retrieval logic
│       ├── llm_server.py  # LLM serving (vLLM/HF)
│       └── rag_pipeline.py # RAG orchestration
├── config_rag.py          # Legacy config (for compatibility)
└── question_answering.py  # Legacy code (deprecated)
```

## Key Differences

### 1. Separation of Concerns

**Old:** Everything in one file - ingestion, retrieval, generation all mixed together.

**New:** Clear separation:
- **Ingestion** (run once): `src/modules/ingest.py`
- **Retrieval** (run many times): `src/modules/retrieve.py`
- **LLM serving**: `src/modules/llm_server.py`
- **Orchestration**: `src/modules/rag_pipeline.py`

### 2. Persistent Storage

**Old:** In-memory storage - need to re-index every time

**New:** ChromaDB persistent storage - index once, query many times

### 3. LLM Serving

**Old:** Basic HuggingFace Transformers

**New:** 
- vLLM for 10x faster inference (with automatic fallback)
- Better quantization support
- Optimized for production use

### 4. Configuration

**Old:** Dictionary-based configuration

**New:** 
- Pydantic-based with validation
- Environment variable support
- Configuration presets (fast/balanced/accurate)
- Type safety

## Migration Steps

### Step 1: Install New Dependencies

```bash
pip install -r requirements.txt
```

Key new dependencies:
- `vllm` - Optimized LLM serving
- `chromadb` - Persistent vector storage
- `pydantic>=2.0.0` - Configuration validation

### Step 2: Migrate Your Code

#### Old Code Pattern:
```python
from question_answering import AdvancedRAGPipeline

pipeline = AdvancedRAGPipeline(
    llm_model_name="Qwen/Qwen2.5-3B-Instruct",
    embedding_model_name="BAAI/bge-small-en-v1.5",
    device="cuda:0",
    persist_dir="./storage"
)

# Ingest documents
pipeline.ingest_documents(md_files)

# Query
answer = pipeline.query("What is a resistor?")
```

#### New Code Pattern:
```python
from src import RAGPipeline, load_config

# Load configuration
config = load_config(preset="balanced")

# Initialize pipeline
pipeline = RAGPipeline(config)

# ONE-TIME: Ingest documents (only run once!)
pipeline.ingest_documents()

# REPEATED: Query (can run many times without re-ingesting)
answer = pipeline.query("What is a resistor?")
```

### Step 3: Update Configuration

#### Old Config (config_rag.py):
```python
LLM_CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "generation_kwargs": {
        "temperature": 0.1,
        "max_new_tokens": 512,
    }
}
```

#### New Config (Pydantic):
```python
from src import RAGConfig, LLMConfig

config = RAGConfig(
    llm=LLMConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        temperature=0.1,
        max_new_tokens=512,
        use_vllm=True,  # NEW: vLLM support
    )
)
```

Or use environment variables:
```bash
export RAG_LLM_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
export RAG_LLM_TEMPERATURE=0.1
```

### Step 4: Separate Ingestion from Querying

This is the biggest change. In the new architecture:

#### Ingestion (Run Once):
```bash
# CLI
python main.py ingest --docs-dir /path/to/documents

# Or programmatically
from src import RAGPipeline, load_config

pipeline = RAGPipeline(load_config())
pipeline.ingest_documents()
```

#### Querying (Run Many Times):
```bash
# CLI
python main.py query --question "What is a resistor?"

# Or programmatically
pipeline = RAGPipeline(load_config())
pipeline.load_index()  # Load existing index
answer = pipeline.query("What is a resistor?")
```

## Feature Mapping

| Old Feature | New Feature | Notes |
|-------------|-------------|-------|
| `AdvancedRAGPipeline` | `RAGPipeline` | Same functionality, cleaner API |
| `ingest_documents()` | `ingest_documents()` | Now persistent, run once |
| `query()` | `query()` | Same API, faster with vLLM |
| `_hybrid_retrieve()` | `HybridRetriever.retrieve()` | Now a separate module |
| `_generate_structured_answer()` | `AnswerGenerator.generate_structured_answer()` | Now a separate module |
| N/A | `load_index()` | NEW: Load existing index |
| N/A | vLLM support | NEW: 10x faster inference |
| N/A | Configuration presets | NEW: fast/balanced/accurate |

## Example Migrations

### Example 1: Simple Query Script

**Old:**
```python
from question_answering import AdvancedRAGPipeline

# Re-index every time (slow!)
pipeline = AdvancedRAGPipeline()
pipeline.ingest_documents(md_files)

# Query
answer = pipeline.query("What is X?")
print(answer.answer)
```

**New:**
```python
from src import RAGPipeline, load_config

# Load existing index (fast!)
pipeline = RAGPipeline(load_config())
pipeline.load_index()

# Query
answer = pipeline.query("What is X?")
print(answer.answer)
```

### Example 2: Batch Processing

**Old:**
```python
# Re-index every time
pipeline = AdvancedRAGPipeline()
pipeline.ingest_documents(md_files)

# Process questions
for question in questions:
    answer = pipeline.query(question)
    print(answer)
```

**New:**
```python
pipeline = RAGPipeline(load_config())
pipeline.load_index()  # Load once

# Batch processing
answers = pipeline.query_batch(questions)
```

### Example 3: Custom Configuration

**Old:**
```python
from config_rag import LLM_CONFIG, EMBEDDING_CONFIG

LLM_CONFIG["model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_CONFIG["model_name"] = "BAAI/bge-large-en-v1.5"

pipeline = AdvancedRAGPipeline(
    llm_model_name=LLM_CONFIG["model_name"],
    embedding_model_name=EMBEDDING_CONFIG["model_name"]
)
```

**New:**
```python
from src import RAGConfig, LLMConfig, EmbeddingConfig

config = RAGConfig(
    llm=LLMConfig(model_name="mistralai/Mistral-7B-Instruct-v0.2"),
    embedding=EmbeddingConfig(model_name="BAAI/bge-large-en-v1.5")
)

pipeline = RAGPipeline(config)
```

## Command-Line Interface

The new system includes a powerful CLI:

```bash
# Ingest documents
python main.py ingest --docs-dir /path/to/documents

# Query single question
python main.py query --question "What is a resistor?"

# Process CSV of questions
python main.py process --questions questions.csv --output answers.csv

# Use different presets
python main.py query --question "..." --preset accurate
```

## Benefits of Migration

1. **Performance**: 
   - 10x faster inference with vLLM
   - No re-indexing needed (persistent storage)
   - Better batching and caching

2. **Maintainability**:
   - Clear module separation
   - Type safety with Pydantic
   - Better error handling
   - Comprehensive logging

3. **Flexibility**:
   - Easy to swap components
   - Configuration presets
   - Environment variable support
   - Multiple storage backends

4. **Production-Ready**:
   - vLLM for high throughput
   - Persistent vector storage
   - Better resource management
   - Comprehensive documentation

## Backward Compatibility

The old code (`question_answering.py` and `config_rag.py`) is still available for backward compatibility, but we recommend migrating to the new modular architecture for better performance and maintainability.

The new `RAGConfig.from_legacy_config()` method can load settings from the old `config_rag.py` file:

```python
from src import RAGConfig

# Load from legacy config
config = RAGConfig.from_legacy_config()
```

## Need Help?

- Check the [README.md](README.md) for detailed documentation
- See [examples/basic_usage.py](examples/basic_usage.py) for code examples
- Open an issue on GitHub for support

## Summary

The new modular architecture provides:
- ✅ Better separation of concerns
- ✅ Persistent storage (no re-indexing)
- ✅ 10x faster inference with vLLM
- ✅ Type-safe configuration
- ✅ Comprehensive CLI
- ✅ Production-ready performance
- ✅ Easy to maintain and extend

Migrate today to take advantage of these improvements!
