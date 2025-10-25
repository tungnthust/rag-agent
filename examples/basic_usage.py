#!/usr/bin/env python3
"""
Example: Basic RAG Pipeline Usage

This script demonstrates how to use the modular RAG system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import RAGPipeline, load_config, setup_logging


def example_basic_usage():
    """Basic usage example"""
    print("="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    # Load configuration
    config = load_config(preset="balanced")
    setup_logging("INFO")
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Step 1: Ingest documents (run once)
    print("\n[Step 1] Ingesting documents...")
    print("Note: This only needs to be done once!")
    
    # In real usage, you would specify your document directory
    # pipeline.ingest_documents()
    
    # Step 2: Load existing index (for subsequent runs)
    print("\n[Step 2] Loading existing index...")
    try:
        pipeline.load_index()
        print("✓ Index loaded successfully")
    except Exception as e:
        print(f"✗ No index found. Please run ingestion first: {e}")
        return
    
    # Step 3: Query the system
    print("\n[Step 3] Querying the system...")
    question = "What is a resistor in electronics?"
    
    print(f"Question: {question}")
    answer = pipeline.query(question)
    
    print(f"\nAnswer: {answer.answer}")
    print(f"\nConfidence: {answer.confidence}")
    
    if answer.citations:
        print("\nCitations:")
        for i, citation in enumerate(answer.citations, 1):
            print(f"  {i}. {citation.document_name} - {citation.section_header}")


def example_with_presets():
    """Example using different presets"""
    print("\n" + "="*80)
    print("Example 2: Using Configuration Presets")
    print("="*80)
    
    presets = ["fast", "balanced", "accurate"]
    
    for preset in presets:
        print(f"\n--- Preset: {preset} ---")
        config = load_config(preset=preset)
        
        print(f"LLM Model: {config.llm.model_name}")
        print(f"Embedding Model: {config.embedding.model_name}")
        print(f"Use vLLM: {config.llm.use_vllm}")
        print(f"Hybrid Top-K: {config.retrieval.hybrid_top_k}")
        print(f"Rerank Top-K: {config.retrieval.rerank_top_k}")


def example_custom_config():
    """Example with custom configuration"""
    print("\n" + "="*80)
    print("Example 3: Custom Configuration")
    print("="*80)
    
    from src import RAGConfig, LLMConfig, EmbeddingConfig, RetrievalConfig
    
    # Create custom configuration
    config = RAGConfig(
        llm=LLMConfig(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            temperature=0.2,
            use_vllm=True,
            max_new_tokens=256,
        ),
        embedding=EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
            device="cuda:0",
        ),
        retrieval=RetrievalConfig(
            hybrid_top_k=30,
            rerank_top_k=5,
            bm25_weight=0.4,
            vector_weight=0.6,
        ),
        use_hyde=True,  # Enable HyDE
        use_decomposition=False,
    )
    
    print("Custom Configuration Created:")
    print(f"  LLM: {config.llm.model_name}")
    print(f"  Temperature: {config.llm.temperature}")
    print(f"  Embedding: {config.embedding.model_name}")
    print(f"  HyDE: {config.use_hyde}")


def example_batch_processing():
    """Example of batch processing"""
    print("\n" + "="*80)
    print("Example 4: Batch Processing")
    print("="*80)
    
    config = load_config(preset="balanced")
    pipeline = RAGPipeline(config)
    
    # Load index
    try:
        pipeline.load_index()
    except Exception as e:
        print(f"✗ No index found. Please run ingestion first: {e}")
        return
    
    # Batch of questions
    questions = [
        "What is a resistor?",
        "How does a capacitor work?",
        "What is Ohm's law?",
    ]
    
    print(f"\nProcessing {len(questions)} questions...")
    
    answers = pipeline.query_batch(questions)
    
    for i, (question, answer) in enumerate(zip(questions, answers), 1):
        print(f"\n{i}. {question}")
        print(f"   Answer: {answer.answer[:100]}...")
        print(f"   Confidence: {answer.confidence}")


def example_programmatic_modules():
    """Example using individual modules"""
    print("\n" + "="*80)
    print("Example 5: Using Individual Modules")
    print("="*80)
    
    from src.modules import HybridRetriever, LLMServer
    
    config = load_config(preset="balanced")
    
    # 1. Use retriever directly
    print("\n[Retriever Module]")
    try:
        retriever = HybridRetriever(config)
        print("✓ Retriever initialized")
        
        # Retrieve documents
        # context_nodes, results = retriever.retrieve("What is a resistor?")
        # print(f"Retrieved {len(context_nodes)} context nodes")
    except Exception as e:
        print(f"✗ Failed to initialize retriever: {e}")
    
    # 2. Use LLM server directly
    print("\n[LLM Server Module]")
    try:
        llm_server = LLMServer(config)
        print("✓ LLM server initialized")
        print(f"Backend: {llm_server.backend}")
        
        # Generate text
        # response = llm_server.generate("Explain what a resistor is")
        # print(f"Response: {response[:100]}...")
    except Exception as e:
        print(f"✗ Failed to initialize LLM server: {e}")


def main():
    """Run all examples"""
    print("\n" + "#"*80)
    print("# RAG System - Usage Examples")
    print("#"*80)
    
    # Run examples
    # example_basic_usage()  # Requires index to be built
    example_with_presets()
    example_custom_config()
    # example_batch_processing()  # Requires index to be built
    example_programmatic_modules()
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)
    print("\nTo run the full system:")
    print("1. First, ingest documents:")
    print("   python main.py ingest --docs-dir /path/to/documents")
    print("\n2. Then, query the system:")
    print("   python main.py query --question 'Your question here'")
    print("="*80)


if __name__ == "__main__":
    main()
