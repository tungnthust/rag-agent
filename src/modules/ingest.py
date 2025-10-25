"""
Ingestion Module

Handles document ingestion and indexing into vector database.
This module is designed to be run once to process and index documents.
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from ..core.config import RAGConfig, IngestionConfig, EmbeddingConfig
from ..core.utils import ensure_dir, find_files

logger = logging.getLogger(__name__)


class MarkdownParserWithMetadata:
    """Parse markdown files into parent-child chunks with header metadata"""
    
    def __init__(self, min_paragraph_length: int = 20):
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.min_paragraph_length = min_paragraph_length
    
    def parse_markdown_file(self, filepath: Path) -> Tuple[List[TextNode], List[TextNode]]:
        """
        Parse a markdown file into parent chunks (sections) and child chunks (paragraphs)
        
        Returns:
            Tuple of (parent_nodes, child_nodes)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get document name from folder and file
        folder_name = filepath.parent.name
        document_name = f"{folder_name}_{filepath.name}"
        
        # Parse sections
        sections = self._parse_sections(content, document_name)
        
        # Create nodes
        return self._create_nodes(sections, document_name)
    
    def _parse_sections(self, content: str, document_name: str) -> List[Dict]:
        """Parse markdown content into sections"""
        sections = []
        current_section = {
            'level': 0,
            'header': document_name,
            'content': '',
            'start_pos': 0
        }
        header_stack = [document_name]
        current_content = []
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            header_match = self.header_pattern.match(line)
            
            if header_match:
                # Save previous section
                if current_content:
                    current_section['content'] = '\n'.join(current_content)
                    sections.append(current_section.copy())
                
                # Start new section
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                
                # Update header stack
                if level <= len(header_stack):
                    header_stack = header_stack[:level]
                header_stack.append(header_text)
                
                current_section = {
                    'level': level,
                    'header': header_text,
                    'content': '',
                    'header_path': ' > '.join(header_stack[1:]),
                    'start_pos': i
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            current_section['content'] = '\n'.join(current_content)
            sections.append(current_section)
        
        return sections
    
    def _create_nodes(self, sections: List[Dict], document_name: str) -> Tuple[List[TextNode], List[TextNode]]:
        """Create parent and child nodes from sections"""
        parent_nodes = []
        child_nodes = []
        
        for idx, section in enumerate(sections):
            if not section['content'].strip():
                continue
            
            # Create parent node (full section)
            parent_id = f"{document_name}_section_{idx}"
            parent_node = TextNode(
                text=section['content'],
                id_=parent_id,
                metadata={
                    'document_name': document_name,
                    'section_header': section['header'],
                    'header_path': section.get('header_path', section['header']),
                    'section_level': section['level'],
                    'node_type': 'parent'
                }
            )
            parent_nodes.append(parent_node)
            
            # Split section into paragraphs for child nodes
            paragraphs = [p.strip() for p in section['content'].split('\n\n') if p.strip()]
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph) < self.min_paragraph_length:
                    continue
                
                child_id = f"{parent_id}_para_{para_idx}"
                child_node = TextNode(
                    text=paragraph,
                    id_=child_id,
                    metadata={
                        'document_name': document_name,
                        'section_header': section['header'],
                        'header_path': section.get('header_path', section['header']),
                        'section_level': section['level'],
                        'node_type': 'child',
                        'parent_id': parent_id
                    },
                    relationships={
                        NodeRelationship.PARENT: RelatedNodeInfo(node_id=parent_id)
                    }
                )
                child_nodes.append(child_node)
        
        return parent_nodes, child_nodes


class DocumentIngestion:
    """
    Document Ingestion Pipeline
    
    Handles ingestion of markdown documents into vector database.
    Should be run once to index all documents.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.markdown_parser = MarkdownParserWithMetadata(
            min_paragraph_length=config.chunking.min_paragraph_length
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.embedding.model_name}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            trust_remote_code=True
        )
        
        # Setup vector store
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Setup ChromaDB vector store"""
        persist_dir = self.config.vector_store.persist_directory
        ensure_dir(persist_dir)
        
        logger.info(f"Initializing ChromaDB at {persist_dir}")
        self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Get or create collection
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.config.vector_store.collection_name
        )
        
        # Setup vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
    
    def ingest_documents(self, document_paths: Optional[List[Path]] = None) -> Tuple[List[TextNode], List[TextNode]]:
        """
        Ingest documents into vector database
        
        Args:
            document_paths: List of paths to markdown files. If None, uses config directory.
            
        Returns:
            Tuple of (parent_nodes, child_nodes)
        """
        # Find documents
        if document_paths is None:
            doc_dir = self.config.ingestion.document_storage_dir
            if not doc_dir.exists():
                raise ValueError(f"Document directory does not exist: {doc_dir}")
            
            document_paths = find_files(doc_dir, f"*{self.config.ingestion.file_extension}")
        
        logger.info(f"Found {len(document_paths)} documents to ingest")
        
        if not document_paths:
            logger.warning("No documents found to ingest")
            return [], []
        
        # Parse documents
        all_parent_nodes = []
        all_child_nodes = []
        
        for filepath in document_paths:
            logger.info(f"Processing: {filepath}")
            try:
                parent_nodes, child_nodes = self.markdown_parser.parse_markdown_file(filepath)
                all_parent_nodes.extend(parent_nodes)
                all_child_nodes.extend(child_nodes)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                continue
        
        logger.info(f"Created {len(all_parent_nodes)} parent chunks and {len(all_child_nodes)} child chunks")
        
        # Index child nodes in vector store
        logger.info("Building vector index...")
        self.index = VectorStoreIndex(
            all_child_nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        # Persist
        logger.info("Persisting index...")
        self.storage_context.persist()
        
        logger.info("Ingestion complete!")
        
        return all_parent_nodes, all_child_nodes
    
    def check_index_exists(self) -> bool:
        """Check if index already exists"""
        try:
            collections = self.chroma_client.list_collections()
            return any(c.name == self.config.vector_store.collection_name for c in collections)
        except Exception:
            return False


def main():
    """Main ingestion function - run this to index documents"""
    from ..core.config import load_config
    from ..core.utils import setup_logging
    
    # Load configuration
    config = load_config()
    setup_logging(config.log_level)
    
    logger.info("Starting document ingestion...")
    
    # Create ingestion pipeline
    ingestion = DocumentIngestion(config)
    
    # Check if index exists
    if ingestion.check_index_exists() and not config.ingestion.force_reindex:
        logger.warning("Index already exists. Use force_reindex=True to rebuild.")
        return
    
    # Ingest documents
    parent_nodes, child_nodes = ingestion.ingest_documents()
    
    logger.info(f"Ingestion complete: {len(parent_nodes)} parent nodes, {len(child_nodes)} child nodes")


if __name__ == "__main__":
    main()
