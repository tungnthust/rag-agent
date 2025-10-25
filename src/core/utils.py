"""
Core Utilities Module

Common utilities and data models used across the RAG system.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import logging
import json
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation information for a source"""
    document_name: str
    section_header: str
    snippet: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class StructuredAnswer:
    """Structured answer with citations"""
    answer: str
    citations: List[Citation]
    confidence: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredAnswer":
        """Create from dictionary"""
        citations = [Citation(**c) for c in data.get("citations", [])]
        return cls(
            answer=data["answer"],
            citations=citations,
            confidence=data.get("confidence")
        )


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files(directory: Path, pattern: str = "*.md") -> List[Path]:
    """Find all files matching pattern in directory"""
    return list(directory.rglob(pattern))


def extract_answer_choices(answer_text: str) -> List[str]:
    """
    Extract answer choices (A, B, C, D) from text.
    
    Args:
        answer_text: Response text
        
    Returns:
        List of answer choices (e.g., ['A'] or ['A', 'B'])
    """
    import re
    
    choices = []
    
    # Pattern 1: "The correct answer is A" or "Answer: A"
    pattern1 = r'\b(?:answer|option|choice)s?\s*(?:is|are|:|-)?\s*([A-D](?:\s*(?:and|,)\s*[A-D])*)'
    matches = re.findall(pattern1, answer_text, re.IGNORECASE)
    
    for match in matches:
        letters = re.findall(r'[A-D]', match.upper())
        choices.extend(letters)
    
    # Pattern 2: Direct mention of letters
    if not choices:
        pattern2 = r'\b([A-D])\s*[:.)\-]'
        matches = re.findall(pattern2, answer_text)
        choices.extend(matches)
    
    # Pattern 3: Look for "correct" near letters
    if not choices:
        pattern3 = r'([A-D])\s+(?:is|are|appears?)\s+correct'
        matches = re.findall(pattern3, answer_text, re.IGNORECASE)
        choices.extend(matches)
    
    # Remove duplicates and sort
    choices = sorted(list(set(choices)))
    
    # If no choices found, look for any capital letters A-D
    if not choices:
        choices = sorted(list(set(re.findall(r'\b([A-D])\b', answer_text))))
    
    return choices if choices else ['A']  # Default to A if nothing found


def format_multiple_choice_question(question: Dict[str, Any]) -> str:
    """
    Format a multiple-choice question for the RAG system.
    
    Args:
        question: Question dictionary with 'question' and 'options' keys
        
    Returns:
        Formatted question string
    """
    formatted = f"{question['question']}\n\n"
    formatted += "Options:\n"
    for key, value in question['options'].items():
        formatted += f"{key}. {value}\n"
    formatted += "\nBased on the provided documents, which option(s) are correct? "
    formatted += "Provide your answer as the letter(s) of the correct option(s)."
    return formatted
