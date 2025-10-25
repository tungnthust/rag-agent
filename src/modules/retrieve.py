"""
Retrieval Module

Handles hybrid search (BM25 + Vector) and reranking for document retrieval.
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

from ..core.config import RAGConfig
from ..core.utils import ensure_dir

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 retriever for keyword-based search"""
    
    def __init__(self, nodes: List[TextNode]):
        self.nodes = nodes
        self.corpus = [node.get_content() for node in nodes]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built with {len(nodes)} nodes")
    
    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[TextNode, float]]:
        """Retrieve top-k nodes using BM25"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.nodes[idx], float(scores[idx])))
        
        return results


class Reranker:
    """Cross-encoder based reranker for fine-grained relevance scoring"""
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(
        self,
        query: str,
        nodes_with_scores: List[Tuple[TextNode, float]],
        top_k: int = 5
    ) -> List[Tuple[TextNode, float]]:
        """Rerank nodes using cross-encoder"""
        if not nodes_with_scores:
            return []
        
        nodes = [item[0] for item in nodes_with_scores]
        texts = [node.get_content() for node in nodes]
        
        # Prepare pairs for cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Combine with nodes and sort
        reranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]


class HybridRetriever:
    """
    Hybrid Retriever combining BM25 and Vector Search
    
    Uses weighted combination of BM25 and dense retrieval for robust search.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.embedding.model_name}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            trust_remote_code=True
        )
        
        # Load vector index
        self._load_vector_index()
        
        # Build BM25 index
        self._build_bm25_index()
        
        # Initialize reranker
        logger.info(f"Loading reranker: {config.reranker.model_name}")
        self.reranker = Reranker(
            model_name=config.reranker.model_name,
            device=config.reranker.device
        )
        
        # Store parent-child mapping
        self.parent_store: Dict[str, TextNode] = {}
    
    def _load_vector_index(self):
        """Load vector index from storage"""
        persist_dir = self.config.vector_store.persist_directory
        
        if not persist_dir.exists():
            raise ValueError(f"Vector store does not exist at {persist_dir}. Run ingestion first.")
        
        logger.info(f"Loading vector index from {persist_dir}")
        
        # Load ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        chroma_collection = chroma_client.get_collection(
            name=self.config.vector_store.collection_name
        )
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Load index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )
        
        logger.info("Vector index loaded successfully")
    
    def _build_bm25_index(self):
        """Build BM25 index from vector store nodes"""
        # Get all nodes from the index
        logger.info("Building BM25 index from stored nodes...")
        
        # Retrieve all nodes (this is a workaround; in production, you'd store BM25 separately)
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10000,  # Large number to get all nodes
        )
        
        # We need to get nodes from the vector store
        # For now, we'll build it on-demand during retrieval
        self.bm25_retriever = None
        logger.warning("BM25 index will be built on first query (requires nodes)")
    
    def set_nodes(self, child_nodes: List[TextNode], parent_nodes: List[TextNode]):
        """
        Set nodes for BM25 and parent-child mapping
        
        This should be called after loading the retriever to set up BM25.
        """
        logger.info(f"Setting up BM25 with {len(child_nodes)} child nodes")
        self.bm25_retriever = BM25Retriever(child_nodes)
        
        # Build parent store
        self.parent_store = {node.id_: node for node in parent_nodes}
        logger.info(f"Parent store built with {len(parent_nodes)} nodes")
    
    def retrieve_hybrid(self, query: str, top_k: int = 50) -> List[Tuple[TextNode, float]]:
        """
        Perform hybrid retrieval (BM25 + Vector)
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (node, score) tuples
        """
        # Vector retrieval
        logger.debug("Performing vector retrieval...")
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        vector_results = retriever.retrieve(query)
        
        # BM25 retrieval (if available)
        bm25_results = []
        if self.bm25_retriever:
            logger.debug("Performing BM25 retrieval...")
            bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        
        # Combine scores
        combined_scores = {}
        
        # Add BM25 scores
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0
            for node, score in bm25_results:
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                combined_scores[node.id_] = (
                    combined_scores.get(node.id_, 0) +
                    normalized_score * self.config.retrieval.bm25_weight
                )
        
        # Add vector scores
        if vector_results:
            for node_with_score in vector_results:
                node_id = node_with_score.node.id_
                score = node_with_score.score if hasattr(node_with_score, 'score') else 1.0
                combined_scores[node_id] = (
                    combined_scores.get(node_id, 0) +
                    score * self.config.retrieval.vector_weight
                )
        
        # Get nodes with combined scores
        node_dict = {}
        if bm25_results:
            node_dict.update({node.id_: node for node, _ in bm25_results})
        for node_with_score in vector_results:
            node_dict[node_with_score.node.id_] = node_with_score.node
        
        results = [(node_dict[node_id], score) for node_id, score in combined_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def rerank(
        self,
        query: str,
        nodes_with_scores: List[Tuple[TextNode, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[TextNode, float]]:
        """
        Rerank retrieved nodes
        
        Args:
            query: Original query
            nodes_with_scores: List of (node, score) tuples
            top_k: Number of top results to return (defaults to config)
            
        Returns:
            Reranked list of (node, score) tuples
        """
        if top_k is None:
            top_k = self.config.retrieval.rerank_top_k
        
        logger.debug(f"Reranking {len(nodes_with_scores)} nodes to top {top_k}")
        return self.reranker.rerank(query, nodes_with_scores, top_k=top_k)
    
    def retrieve_parent_chunks(self, child_nodes: List[TextNode]) -> List[TextNode]:
        """
        Retrieve parent chunks from child nodes (small-to-big strategy)
        
        Args:
            child_nodes: List of child nodes
            
        Returns:
            List of parent nodes
        """
        parent_chunks = []
        seen_parents = set()
        
        for child_node in child_nodes:
            parent_id = child_node.metadata.get('parent_id')
            if parent_id and parent_id not in seen_parents and parent_id in self.parent_store:
                parent_chunks.append(self.parent_store[parent_id])
                seen_parents.add(parent_id)
        
        logger.debug(f"Retrieved {len(parent_chunks)} unique parent chunks")
        return parent_chunks
    
    def retrieve(
        self,
        query: str,
        return_parent: bool = True
    ) -> Tuple[List[TextNode], List[Tuple[TextNode, float]]]:
        """
        Full retrieval pipeline: hybrid search -> reranking -> parent retrieval
        
        Args:
            query: Search query
            return_parent: Whether to return parent chunks (small-to-big)
            
        Returns:
            Tuple of (context_nodes, reranked_results)
        """
        # Hybrid retrieval
        logger.info(f"Retrieving for query: {query[:100]}...")
        hybrid_results = self.retrieve_hybrid(
            query,
            top_k=self.config.retrieval.hybrid_top_k
        )
        
        # Reranking
        reranked_results = self.rerank(query, hybrid_results)
        
        # Get child nodes
        child_nodes = [node for node, _ in reranked_results]
        
        # Get parent chunks if requested
        if return_parent and self.parent_store:
            context_nodes = self.retrieve_parent_chunks(child_nodes)
        else:
            context_nodes = child_nodes
        
        return context_nodes, reranked_results


def main():
    """Test retrieval functionality"""
    from ..core.config import load_config
    from ..core.utils import setup_logging
    
    config = load_config()
    setup_logging(config.log_level)
    
    logger.info("Testing retrieval module...")
    
    # Initialize retriever
    retriever = HybridRetriever(config)
    
    # Test query
    test_query = "What is a resistor?"
    context_nodes, results = retriever.retrieve(test_query)
    
    logger.info(f"Retrieved {len(context_nodes)} context nodes")
    for i, (node, score) in enumerate(results, 1):
        logger.info(f"{i}. Score: {score:.4f}, Text: {node.get_content()[:100]}...")


if __name__ == "__main__":
    main()
