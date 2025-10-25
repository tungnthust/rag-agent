"""Core utilities and configuration for the RAG system"""

from .config import RAGConfig, load_config, PRESETS
from .utils import (
    Citation,
    StructuredAnswer,
    setup_logging,
    extract_answer_choices,
    format_multiple_choice_question,
    ensure_dir,
    find_files
)

__all__ = [
    "RAGConfig",
    "load_config",
    "PRESETS",
    "Citation",
    "StructuredAnswer",
    "setup_logging",
    "extract_answer_choices",
    "format_multiple_choice_question",
    "ensure_dir",
    "find_files",
]
