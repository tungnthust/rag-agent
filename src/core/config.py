"""
Core Configuration Module

This module centralizes all configuration for the RAG system.
Uses Pydantic for validation and environment variable support.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
from pathlib import Path


class LLMConfig(BaseModel):
    """Configuration for Language Model"""
    model_name: str = Field(default="Qwen/Qwen2.5-3B-Instruct")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=512, gt=0)
    use_vllm: bool = Field(default=True, description="Use vLLM for serving")
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, ge=0.0, le=1.0)
    quantization: Optional[str] = Field(default=None, description="awq, gptq, or None")


class EmbeddingConfig(BaseModel):
    """Configuration for Embedding Model"""
    model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    device: str = Field(default="cuda:0")
    batch_size: int = Field(default=32, gt=0)


class RerankerConfig(BaseModel):
    """Configuration for Reranker Model"""
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    device: str = Field(default="cuda:0")


class RetrievalConfig(BaseModel):
    """Configuration for Retrieval"""
    hybrid_top_k: int = Field(default=50, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.0, ge=0.0)


class ChunkingConfig(BaseModel):
    """Configuration for Document Chunking"""
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    min_paragraph_length: int = Field(default=20, gt=0)


class VectorStoreConfig(BaseModel):
    """Configuration for Vector Store"""
    provider: str = Field(default="chroma", description="chroma or faiss")
    persist_directory: Path = Field(default=Path("./storage/vector_db"))
    collection_name: str = Field(default="rag_documents")


class IngestionConfig(BaseModel):
    """Configuration for Document Ingestion"""
    document_storage_dir: Path = Field(default=Path("./documents"))
    file_extension: str = Field(default=".md")
    batch_size: int = Field(default=10, gt=0)
    force_reindex: bool = Field(default=False)


class RAGConfig(BaseModel):
    """Main RAG Configuration"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    
    # Query processing
    use_hyde: bool = Field(default=True)
    use_decomposition: bool = Field(default=False)
    
    # Logging
    log_level: str = Field(default="INFO")
    verbose: bool = Field(default=True)

    class Config:
        env_prefix = "RAG_"
        
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls()
    
    @classmethod
    def from_legacy_config(cls):
        """Load from legacy config_rag.py for backward compatibility"""
        try:
            from config_rag import (
                LLM_CONFIG, EMBEDDING_CONFIG, RERANKER_CONFIG,
                RETRIEVAL_CONFIG, CHUNKING_CONFIG, INGESTION_CONFIG,
                QUERY_CONFIG
            )
            
            return cls(
                llm=LLMConfig(
                    model_name=LLM_CONFIG["model_name"],
                    temperature=LLM_CONFIG["generation_kwargs"]["temperature"],
                    top_p=LLM_CONFIG["generation_kwargs"]["top_p"],
                    max_new_tokens=LLM_CONFIG["generation_kwargs"]["max_new_tokens"],
                ),
                embedding=EmbeddingConfig(
                    model_name=EMBEDDING_CONFIG["model_name"],
                ),
                reranker=RerankerConfig(
                    model_name=RERANKER_CONFIG["model_name"],
                ),
                retrieval=RetrievalConfig(
                    hybrid_top_k=RETRIEVAL_CONFIG["hybrid_top_k"],
                    rerank_top_k=RETRIEVAL_CONFIG["rerank_top_k"],
                    bm25_weight=RETRIEVAL_CONFIG["bm25_weight"],
                    vector_weight=RETRIEVAL_CONFIG["vector_weight"],
                ),
                chunking=ChunkingConfig(
                    chunk_size=CHUNKING_CONFIG["chunk_size"],
                    chunk_overlap=CHUNKING_CONFIG["chunk_overlap"],
                    min_paragraph_length=CHUNKING_CONFIG["min_paragraph_length"],
                ),
                ingestion=IngestionConfig(
                    document_storage_dir=Path(INGESTION_CONFIG["document_storage_dir"]),
                    file_extension=INGESTION_CONFIG["file_extension"],
                    batch_size=INGESTION_CONFIG["batch_size"],
                ),
                use_hyde=QUERY_CONFIG["use_hyde"],
                use_decomposition=QUERY_CONFIG["use_decomposition"],
            )
        except ImportError:
            return cls()


# Presets for different use cases
PRESETS = {
    "fast": RAGConfig(
        llm=LLMConfig(model_name="microsoft/phi-2", use_vllm=False),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        retrieval=RetrievalConfig(hybrid_top_k=30, rerank_top_k=3),
    ),
    "balanced": RAGConfig(
        llm=LLMConfig(model_name="Qwen/Qwen2.5-3B-Instruct"),
        embedding=EmbeddingConfig(model_name="BAAI/bge-small-en-v1.5"),
        retrieval=RetrievalConfig(hybrid_top_k=50, rerank_top_k=5),
    ),
    "accurate": RAGConfig(
        llm=LLMConfig(model_name="mistralai/Mistral-7B-Instruct-v0.2"),
        embedding=EmbeddingConfig(model_name="BAAI/bge-large-en-v1.5"),
        retrieval=RetrievalConfig(hybrid_top_k=100, rerank_top_k=10),
    ),
}


def load_config(preset: Optional[str] = None, config_path: Optional[str] = None) -> RAGConfig:
    """
    Load configuration from various sources
    
    Args:
        preset: Name of preset configuration (fast, balanced, accurate)
        config_path: Path to JSON/YAML config file
        
    Returns:
        RAGConfig instance
    """
    if preset and preset in PRESETS:
        return PRESETS[preset]
    
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config_dict = json.load(f)
        return RAGConfig(**config_dict)
    
    # Try loading from legacy config
    return RAGConfig.from_legacy_config()
