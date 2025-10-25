#!/usr/bin/env python3
"""
Main Entry Point for RAG Agent

Usage:
    # Ingest documents (one-time operation)
    python main.py ingest --docs-dir /path/to/documents
    
    # Query the system
    python main.py query --question "What is a resistor?"
    
    # Process questions from CSV
    python main.py process --questions questions.csv --output answers.csv
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    RAGPipeline,
    load_config,
    setup_logging,
    extract_answer_choices,
    format_multiple_choice_question
)

logger = logging.getLogger(__name__)


def ingest_command(args):
    """Handle document ingestion"""
    logger.info("Starting document ingestion...")
    
    # Load config
    config = load_config(preset=args.preset)
    
    # Override docs directory if provided
    if args.docs_dir:
        config.ingestion.document_storage_dir = Path(args.docs_dir)
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Ingest documents
    parent_nodes, child_nodes = pipeline.ingest_documents(
        force_reindex=args.force_reindex
    )
    
    logger.info(f"✓ Ingestion complete!")
    logger.info(f"  - Parent nodes: {len(parent_nodes)}")
    logger.info(f"  - Child nodes: {len(child_nodes)}")


def query_command(args):
    """Handle single query"""
    logger.info("Processing query...")
    
    # Load config
    config = load_config(preset=args.preset)
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Load index
    pipeline.load_index()
    
    # Query
    answer = pipeline.query(
        args.question,
        use_hyde=not args.no_hyde,
        use_decomposition=args.decompose
    )
    
    # Print results
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(answer.answer)
    
    if answer.citations:
        print("\n" + "="*80)
        print("CITATIONS:")
        print("="*80)
        for i, citation in enumerate(answer.citations, 1):
            print(f"\n{i}. {citation.document_name}")
            print(f"   Section: {citation.section_header}")
            print(f"   Snippet: {citation.snippet[:150]}...")
    
    print("\n" + "="*80)
    print(f"Confidence: {answer.confidence}")
    print("="*80)


def process_command(args):
    """Handle batch processing from CSV"""
    logger.info("Processing questions from CSV...")
    
    # Load config
    config = load_config(preset=args.preset)
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Load index
    pipeline.load_index()
    
    # Load questions
    questions_df = pd.read_csv(args.questions)
    logger.info(f"Loaded {len(questions_df)} questions")
    
    # Validate CSV format
    required_columns = ['Question', 'A', 'B', 'C', 'D']
    for col in required_columns:
        if col not in questions_df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    
    # Process questions
    answers = []
    
    for idx, row in questions_df.iterrows():
        question_id = idx + 1
        logger.info(f"Processing question {question_id}/{len(questions_df)}")
        
        # Format question
        question_dict = {
            'question': row['Question'],
            'options': {
                'A': row['A'],
                'B': row['B'],
                'C': row['C'],
                'D': row['D'],
            }
        }
        
        formatted_question = format_multiple_choice_question(question_dict)
        
        try:
            # Query
            result = pipeline.query(
                formatted_question,
                use_hyde=not args.no_hyde,
                use_decomposition=args.decompose
            )
            
            # Extract answer choices
            choices = extract_answer_choices(result.answer)
            
            # Format answer
            if len(choices) == 1:
                answer_str = choices[0]
            else:
                answer_str = ",".join(choices)
            
            answers.append((question_id, answer_str))
            
            logger.info(f"  Answer: {answer_str} (confidence: {result.confidence})")
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            answers.append((question_id, "A"))  # Default to A on error
    
    # Save answers
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for question_id, answer in answers:
            if ',' in answer:
                f.write(f'{question_id},"{answer}"\n')
            else:
                f.write(f'{question_id},{answer}\n')
    
    logger.info(f"✓ Answers saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Agent - Modular RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument("--preset", choices=["fast", "balanced", "accurate"],
                       help="Configuration preset")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest",
                                         help="Ingest documents into vector database")
    ingest_parser.add_argument("--docs-dir", type=str,
                              help="Directory containing markdown documents")
    ingest_parser.add_argument("--force-reindex", action="store_true",
                              help="Force reindexing even if index exists")
    
    # Query command
    query_parser = subparsers.add_parser("query",
                                        help="Query the RAG system")
    query_parser.add_argument("--question", type=str, required=True,
                             help="Question to ask")
    query_parser.add_argument("--no-hyde", action="store_true",
                             help="Disable HyDE query enhancement")
    query_parser.add_argument("--decompose", action="store_true",
                             help="Enable query decomposition")
    
    # Process command
    process_parser = subparsers.add_parser("process",
                                          help="Process questions from CSV")
    process_parser.add_argument("--questions", type=str, required=True,
                               help="Path to questions CSV file")
    process_parser.add_argument("--output", type=str, default="answers.csv",
                               help="Path to output answers CSV")
    process_parser.add_argument("--no-hyde", action="store_true",
                               help="Disable HyDE query enhancement")
    process_parser.add_argument("--decompose", action="store_true",
                               help="Enable query decomposition")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "query":
        query_command(args)
    elif args.command == "process":
        process_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
