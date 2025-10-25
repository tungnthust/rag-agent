"""
RAG Agent - Modular RAG System

A modular, production-ready RAG system with:
- Document ingestion with parent-child chunking
- Hybrid search (BM25 + Vector)
- Reranking with cross-encoders
- Optimized LLM serving with vLLM
- Query preprocessing (HyDE, decomposition)
- Structured outputs with citations
"""

__version__ = "2.0.0"

from .core.config import RAGConfig, load_config, PRESETS
from .core.utils import (
    Citation,
    StructuredAnswer,
    setup_logging,
    extract_answer_choices,
    format_multiple_choice_question
)
from .modules.ingest import DocumentIngestion
from .modules.retrieve import HybridRetriever
from .modules.llm_server import LLMServer, QueryPreprocessor, AnswerGenerator
from .modules.rag_pipeline import RAGPipeline

__all__ = [
    # Config
    "RAGConfig",
    "load_config",
    "PRESETS",
    
    # Utils
    "Citation",
    "StructuredAnswer",
    "setup_logging",
    "extract_answer_choices",
    "format_multiple_choice_question",
    
    # Modules
    "DocumentIngestion",
    "HybridRetriever",
    "LLMServer",
    "QueryPreprocessor",
    "AnswerGenerator",
    "RAGPipeline",
]
