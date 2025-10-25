"""
RAG Pipeline Module

Orchestrates the complete RAG pipeline with ingestion, retrieval, and generation.
"""

import logging
from typing import List, Optional, Tuple
from pathlib import Path

from llama_index.core.schema import TextNode

from ..core.config import RAGConfig, load_config
from ..core.utils import StructuredAnswer, setup_logging
from .ingest import DocumentIngestion
from .retrieve import HybridRetriever
from .llm_server import LLMServer, QueryPreprocessor, AnswerGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG Pipeline
    
    Orchestrates document ingestion, retrieval, and answer generation.
    Separates one-time ingestion from repeated retrieval/generation.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG Pipeline
        
        Args:
            config: Configuration object. If None, loads from default sources.
        """
        if config is None:
            config = load_config()
        
        self.config = config
        setup_logging(config.log_level)
        
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize components (lazy loading for efficiency)
        self._ingestion: Optional[DocumentIngestion] = None
        self._retriever: Optional[HybridRetriever] = None
        self._llm_server: Optional[LLMServer] = None
        self._query_preprocessor: Optional[QueryPreprocessor] = None
        self._answer_generator: Optional[AnswerGenerator] = None
        
        # Storage for nodes
        self._parent_nodes: List[TextNode] = []
        self._child_nodes: List[TextNode] = []
    
    @property
    def ingestion(self) -> DocumentIngestion:
        """Lazy load ingestion module"""
        if self._ingestion is None:
            self._ingestion = DocumentIngestion(self.config)
        return self._ingestion
    
    @property
    def retriever(self) -> HybridRetriever:
        """Lazy load retriever module"""
        if self._retriever is None:
            self._retriever = HybridRetriever(self.config)
        return self._retriever
    
    @property
    def llm_server(self) -> LLMServer:
        """Lazy load LLM server"""
        if self._llm_server is None:
            self._llm_server = LLMServer(self.config)
        return self._llm_server
    
    @property
    def query_preprocessor(self) -> QueryPreprocessor:
        """Lazy load query preprocessor"""
        if self._query_preprocessor is None:
            self._query_preprocessor = QueryPreprocessor(self.llm_server)
        return self._query_preprocessor
    
    @property
    def answer_generator(self) -> AnswerGenerator:
        """Lazy load answer generator"""
        if self._answer_generator is None:
            self._answer_generator = AnswerGenerator(self.llm_server)
        return self._answer_generator
    
    def ingest_documents(
        self,
        document_paths: Optional[List[Path]] = None,
        force_reindex: bool = False
    ) -> Tuple[List[TextNode], List[TextNode]]:
        """
        Ingest documents into vector database (run once)
        
        Args:
            document_paths: List of document paths. If None, uses config directory.
            force_reindex: Force reindexing even if index exists
            
        Returns:
            Tuple of (parent_nodes, child_nodes)
        """
        logger.info("Starting document ingestion...")
        
        # Check if already indexed
        if not force_reindex and self.ingestion.check_index_exists():
            logger.info("Index already exists. Skipping ingestion.")
            logger.info("Use force_reindex=True to rebuild the index.")
            return self._parent_nodes, self._child_nodes
        
        # Ingest documents
        parent_nodes, child_nodes = self.ingestion.ingest_documents(document_paths)
        
        # Store nodes
        self._parent_nodes = parent_nodes
        self._child_nodes = child_nodes
        
        logger.info(f"Ingestion complete: {len(parent_nodes)} parent nodes, {len(child_nodes)} child nodes")
        
        return parent_nodes, child_nodes
    
    def load_index(self):
        """
        Load existing index and setup retriever
        
        Call this before querying if you've already run ingestion.
        """
        logger.info("Loading existing index...")
        
        # Initialize retriever (this loads the index)
        _ = self.retriever
        
        logger.info("Index loaded successfully")
    
    def setup_retriever_nodes(
        self,
        child_nodes: Optional[List[TextNode]] = None,
        parent_nodes: Optional[List[TextNode]] = None
    ):
        """
        Setup retriever with nodes for BM25 and parent-child mapping
        
        Args:
            child_nodes: Child nodes for BM25 indexing
            parent_nodes: Parent nodes for small-to-big retrieval
        """
        if child_nodes is None:
            child_nodes = self._child_nodes
        if parent_nodes is None:
            parent_nodes = self._parent_nodes
        
        if not child_nodes or not parent_nodes:
            logger.warning("No nodes provided to retriever. BM25 and parent retrieval will not work.")
            return
        
        self.retriever.set_nodes(child_nodes, parent_nodes)
        logger.info("Retriever nodes configured")
    
    def query(
        self,
        question: str,
        use_hyde: Optional[bool] = None,
        use_decomposition: Optional[bool] = None,
        include_citations: bool = True
    ) -> StructuredAnswer:
        """
        Query the RAG pipeline
        
        Args:
            question: User question
            use_hyde: Use HyDE for query enhancement (defaults to config)
            use_decomposition: Use query decomposition (defaults to config)
            include_citations: Include citations in answer
            
        Returns:
            StructuredAnswer with answer and citations
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        # Use config defaults if not specified
        if use_hyde is None:
            use_hyde = self.config.use_hyde
        if use_decomposition is None:
            use_decomposition = self.config.use_decomposition
        
        # Query preprocessing
        queries = [question]
        if use_decomposition:
            logger.info("Decomposing query...")
            queries = self.query_preprocessor.decompose_query(question)
            logger.info(f"Sub-queries: {queries}")
        
        all_retrieved_nodes = []
        
        for query in queries:
            search_query = query
            
            if use_hyde:
                logger.info("Generating hypothetical document...")
                hyde_doc = self.query_preprocessor.generate_hypothetical_document(query)
                search_query = f"{query} {hyde_doc[:200]}"  # Combine query with HyDE
            
            # Retrieve
            context_nodes, reranked_results = self.retriever.retrieve(search_query)
            all_retrieved_nodes.extend(context_nodes)
        
        # Remove duplicates
        unique_nodes = []
        seen_ids = set()
        for node in all_retrieved_nodes:
            if node.id_ not in seen_ids:
                unique_nodes.append(node)
                seen_ids.add(node.id_)
        
        # Limit to top nodes
        unique_nodes = unique_nodes[:self.config.retrieval.rerank_top_k]
        
        # Generate answer
        logger.info("Generating structured answer...")
        answer = self.answer_generator.generate_structured_answer(
            question,
            unique_nodes,
            include_citations=include_citations
        )
        
        return answer
    
    def query_batch(
        self,
        questions: List[str],
        **query_kwargs
    ) -> List[StructuredAnswer]:
        """
        Query multiple questions in batch
        
        Args:
            questions: List of questions
            **query_kwargs: Arguments passed to query()
            
        Returns:
            List of StructuredAnswer objects
        """
        logger.info(f"Processing batch of {len(questions)} questions...")
        
        answers = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Question {i}/{len(questions)}")
            try:
                answer = self.query(question, **query_kwargs)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                # Return empty answer on error
                from ..core.utils import Citation
                answers.append(StructuredAnswer(
                    answer="Error processing question",
                    citations=[],
                    confidence="low"
                ))
        
        logger.info("Batch processing complete")
        return answers


def main():
    """Main entry point for RAG pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--mode", choices=["ingest", "query"], required=True,
                      help="Mode: ingest documents or query")
    parser.add_argument("--docs-dir", type=str,
                      help="Directory containing markdown documents (for ingest)")
    parser.add_argument("--query", type=str,
                      help="Query string (for query mode)")
    parser.add_argument("--preset", choices=["fast", "balanced", "accurate"],
                      help="Configuration preset")
    parser.add_argument("--force-reindex", action="store_true",
                      help="Force reindexing even if index exists")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(preset=args.preset)
    
    # Override docs directory if provided
    if args.docs_dir:
        config.ingestion.document_storage_dir = Path(args.docs_dir)
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    if args.mode == "ingest":
        # Ingest documents
        logger.info("Starting document ingestion...")
        parent_nodes, child_nodes = pipeline.ingest_documents(
            force_reindex=args.force_reindex
        )
        logger.info(f"Ingested {len(parent_nodes)} parent nodes, {len(child_nodes)} child nodes")
        
    elif args.mode == "query":
        if not args.query:
            logger.error("--query is required for query mode")
            return
        
        # Load index and query
        logger.info("Loading index...")
        pipeline.load_index()
        
        # Query
        logger.info(f"Querying: {args.query}")
        answer = pipeline.query(args.query)
        
        # Print results
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer.answer)
        print("\n" + "="*80)
        print("CITATIONS:")
        print("="*80)
        for i, citation in enumerate(answer.citations, 1):
            print(f"\n{i}. {citation.document_name} - {citation.section_header}")
            print(f"   {citation.snippet[:200]}...")
        print("\n" + "="*80)
        print(f"Confidence: {answer.confidence}")
        print("="*80)


if __name__ == "__main__":
    main()
