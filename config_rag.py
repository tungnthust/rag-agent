"""
Configuration file for Advanced RAG Pipeline

This file contains all configurable parameters for the pipeline.
Modify these settings to customize the behavior.
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# LLM Configuration
LLM_CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",  # Main LLM for generation
    "alternatives": [
        "microsoft/phi-2",                      # Fast, efficient
        "mistralai/Mistral-7B-Instruct-v0.2",  # Higher quality
        "meta-llama/Llama-2-7b-chat-hf",       # Good balance
        "google/gemma-2b-it",                   # Small, efficient
    ],
    "generation_kwargs": {
        "temperature": 0.1,      # Lower = more deterministic
        "top_p": 0.95,          # Nucleus sampling threshold
        "max_new_tokens": 512,   # Maximum length of generated text
        "do_sample": True,       # Enable sampling
    },
    "quantization": {
        "load_in_4bit": True,           # Use 4-bit quantization
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
}

# Embedding Model Configuration
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-small-en-v1.5",  # Multilingual, high quality
    "alternatives": [
        "BAAI/bge-large-en-v1.5",              # English-only, high quality
        "BAAI/bge-base-en-v1.5",               # Faster, good quality
        "sentence-transformers/all-MiniLM-L6-v2",  # Very fast
        "intfloat/multilingual-e5-large",      # Multilingual alternative
    ],
    "trust_remote_code": True,
}

# Reranker Configuration
RERANKER_CONFIG = {
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "alternatives": [
        "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Larger, more accurate
        "BAAI/bge-reranker-base",                  # BGE reranker
        "BAAI/bge-reranker-large",                 # Larger BGE reranker
    ],
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE_CONFIG = {
    "device": "cuda:0",      # GPU device (cuda:0, cuda:1, or cpu)
    "gpu_id": 0,             # GPU ID for models that need it
    "use_gpu": True,         # Whether to use GPU
}

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

RETRIEVAL_CONFIG = {
    "hybrid_top_k": 50,          # Number of candidates from hybrid search
    "rerank_top_k": 5,           # Number after reranking
    "bm25_weight": 0.5,          # Weight for BM25 scores (0-1)
    "vector_weight": 0.5,        # Weight for vector scores (0-1)
    "similarity_threshold": 0.0,  # Minimum similarity score
}

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

CHUNKING_CONFIG = {
    "chunk_size": 512,           # Maximum chunk size in tokens
    "chunk_overlap": 50,         # Overlap between chunks
    "min_paragraph_length": 20,  # Minimum paragraph length to include
}

# ============================================================================
# QUERY PREPROCESSING CONFIGURATION
# ============================================================================

QUERY_CONFIG = {
    "use_hyde": True,               # Use HyDE by default
    "use_decomposition": False,     # Use query decomposition by default
    "hyde_doc_length": 200,         # Length of hypothetical document
    "max_sub_queries": 3,           # Maximum number of sub-queries
}

# ============================================================================
# DOCUMENT INGESTION CONFIGURATION
# ============================================================================

INGESTION_CONFIG = {
    "document_storage_dir": "/home/public/hoangnguyen/TechnicalDocument/Solution/extraction/submission",
    "persist_dir": "./storage",
    "file_extension": ".md",
    "batch_size": 10,  # Process documents in batches
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    "questions_csv": "questions.csv",
    "answers_output": "answers.txt",
    "structured_output_file": "structured_answers.json",
    "include_confidence": True,
    "include_citations": True,
    "max_citations": 5,
}

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

PERFORMANCE_CONFIG = {
    "use_caching": True,         # Cache embeddings
    "batch_encode": True,        # Batch encoding for embeddings
    "lazy_load": False,          # Load models on demand
    "num_workers": 4,            # Number of workers for parallel processing
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "log_level": "INFO",         # DEBUG, INFO, WARNING, ERROR
    "log_file": "rag_pipeline.log",
    "verbose": True,             # Print detailed progress
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_llm_config():
    """Get LLM configuration"""
    return LLM_CONFIG.copy()

def get_embedding_config():
    """Get embedding configuration"""
    return EMBEDDING_CONFIG.copy()

def get_reranker_config():
    """Get reranker configuration"""
    return RERANKER_CONFIG.copy()

def get_device():
    """Get device configuration"""
    import torch
    if DEVICE_CONFIG["use_gpu"] and torch.cuda.is_available():
        return DEVICE_CONFIG["device"]
    return "cpu"

def update_config(config_dict: dict, updates: dict):
    """Update configuration with custom values"""
    config_dict.update(updates)
    return config_dict

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

PRESETS = {
    "fast": {
        "llm_model": "microsoft/phi-2",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "hybrid_top_k": 30,
        "rerank_top_k": 3,
    },
    "balanced": {
        "llm_model": "Qwen/Qwen2.5-3B-Instruct",
        "embedding_model": "BAAI/bge-m3",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "hybrid_top_k": 50,
        "rerank_top_k": 5,
    },
    "accurate": {
        "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "hybrid_top_k": 100,
        "rerank_top_k": 10,
    },
}

def load_preset(preset_name: str):
    """Load a configuration preset"""
    if preset_name not in PRESETS:
        raise ValueError(f"Preset '{preset_name}' not found. Available: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    
    # Update configurations
    if "llm_model" in preset:
        LLM_CONFIG["model_name"] = preset["llm_model"]
    if "embedding_model" in preset:
        EMBEDDING_CONFIG["model_name"] = preset["embedding_model"]
    if "reranker_model" in preset:
        RERANKER_CONFIG["model_name"] = preset["reranker_model"]
    if "hybrid_top_k" in preset:
        RETRIEVAL_CONFIG["hybrid_top_k"] = preset["hybrid_top_k"]
    if "rerank_top_k" in preset:
        RETRIEVAL_CONFIG["rerank_top_k"] = preset["rerank_top_k"]
    
    return preset

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("RAG Pipeline Configuration")
    print("=" * 80)
    
    print(f"\nLLM Model: {LLM_CONFIG['model_name']}")
    print(f"Embedding Model: {EMBEDDING_CONFIG['model_name']}")
    print(f"Reranker Model: {RERANKER_CONFIG['model_name']}")
    print(f"Device: {get_device()}")
    
    print(f"\nRetrieval Settings:")
    print(f"  Hybrid Top-K: {RETRIEVAL_CONFIG['hybrid_top_k']}")
    print(f"  Rerank Top-K: {RETRIEVAL_CONFIG['rerank_top_k']}")
    
    print(f"\nAvailable Presets:")
    for preset_name in PRESETS.keys():
        print(f"  - {preset_name}")
    
    print("\n" + "=" * 80)
    print("To use a preset, call: load_preset('balanced')")
    print("To change individual settings, modify the CONFIG dictionaries")