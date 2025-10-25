"""
LLM Serving Module

Provides optimized LLM inference using vLLM or HuggingFace Transformers.
vLLM is preferred for production use due to its optimized serving.
"""

import logging
import json
import re
from typing import List, Optional, Union
from llama_index.core.schema import TextNode

from ..core.config import RAGConfig, LLMConfig
from ..core.utils import Citation, StructuredAnswer

logger = logging.getLogger(__name__)


class LLMServer:
    """
    LLM Server for inference
    
    Supports both vLLM (optimized) and HuggingFace Transformers (fallback).
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm_config = config.llm
        
        # Initialize LLM
        if self.llm_config.use_vllm:
            try:
                self._init_vllm()
            except Exception as e:
                logger.warning(f"Failed to initialize vLLM: {e}. Falling back to HuggingFace.")
                self._init_huggingface()
        else:
            self._init_huggingface()
    
    def _init_vllm(self):
        """Initialize vLLM for optimized inference"""
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Initializing vLLM with model: {self.llm_config.model_name}")
            
            self.llm = LLM(
                model=self.llm_config.model_name,
                tensor_parallel_size=self.llm_config.tensor_parallel_size,
                gpu_memory_utilization=self.llm_config.gpu_memory_utilization,
                trust_remote_code=True,
                quantization=self.llm_config.quantization,
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.llm_config.temperature,
                top_p=self.llm_config.top_p,
                max_tokens=self.llm_config.max_new_tokens,
            )
            
            self.backend = "vllm"
            logger.info("vLLM initialized successfully")
            
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    def _init_huggingface(self):
        """Initialize HuggingFace Transformers (fallback)"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        logger.info(f"Initializing HuggingFace Transformers with model: {self.llm_config.model_name}")
        
        # Setup quantization if needed
        quantization_config = None
        if self.llm_config.quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_config.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.backend = "huggingface"
        logger.info("HuggingFace Transformers initialized successfully")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.backend == "vllm":
            return self._generate_vllm(prompt, **kwargs)
        else:
            return self._generate_huggingface(prompt, **kwargs)
    
    def _generate_vllm(self, prompt: str, **kwargs) -> str:
        """Generate using vLLM"""
        from vllm import SamplingParams
        
        # Override sampling params if provided
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', self.llm_config.temperature),
            top_p=kwargs.get('top_p', self.llm_config.top_p),
            max_tokens=kwargs.get('max_tokens', self.llm_config.max_new_tokens),
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def _generate_huggingface(self, prompt: str, **kwargs) -> str:
        """Generate using HuggingFace Transformers"""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', self.llm_config.max_new_tokens),
                temperature=kwargs.get('temperature', self.llm_config.temperature),
                top_p=kwargs.get('top_p', self.llm_config.top_p),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part (not the input)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Alias for generate() for LlamaIndex compatibility"""
        return self.generate(prompt, **kwargs)


class QueryPreprocessor:
    """Preprocess queries with decomposition and HyDE"""
    
    def __init__(self, llm_server: LLMServer):
        self.llm = llm_server
    
    def decompose_query(self, query: str, max_sub_queries: int = 3) -> List[str]:
        """
        Decompose complex query into sub-queries
        
        Args:
            query: Original query
            max_sub_queries: Maximum number of sub-queries
            
        Returns:
            List of sub-queries
        """
        prompt = f"""Given the following question, break it down into simpler sub-questions if it's complex. 
If the question is already simple, return it as is.

Question: {query}

Sub-questions (one per line, max {max_sub_queries}):"""
        
        response = self.llm.generate(prompt, max_tokens=200)
        
        # Parse sub-queries
        sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-')]
        
        # If no sub-queries found, return original
        if not sub_queries:
            return [query]
        
        return sub_queries[:max_sub_queries]
    
    def generate_hypothetical_document(self, query: str, doc_length: int = 200) -> str:
        """
        Generate a hypothetical document for HyDE
        
        Args:
            query: Original query
            doc_length: Desired length of hypothetical document
            
        Returns:
            Hypothetical document text
        """
        prompt = f"""Generate a detailed answer to the following question as if you were writing documentation:

Question: {query}

Answer:"""
        
        response = self.llm.generate(prompt, max_tokens=doc_length)
        return response


class AnswerGenerator:
    """Generate structured answers with citations"""
    
    def __init__(self, llm_server: LLMServer):
        self.llm = llm_server
    
    def generate_structured_answer(
        self,
        query: str,
        context_nodes: List[TextNode],
        include_citations: bool = True,
        max_citations: int = 5
    ) -> StructuredAnswer:
        """
        Generate structured answer with citations
        
        Args:
            query: User query
            context_nodes: Retrieved context nodes
            include_citations: Whether to include citations
            max_citations: Maximum number of citations
            
        Returns:
            StructuredAnswer with answer and citations
        """
        # Prepare context
        context_text = "\n\n---\n\n".join([
            f"Source {i+1}:\nDocument: {node.metadata.get('document_name', 'Unknown')}\n"
            f"Section: {node.metadata.get('header_path', 'Unknown')}\n"
            f"Content: {node.get_content()}"
            for i, node in enumerate(context_nodes[:max_citations])
        ])
        
        # Create prompt for structured output
        prompt = f"""Based on the following context, answer the question and provide citations.

Context:
{context_text}

Question: {query}

Provide your answer in the following JSON format:
{{
    "answer": "Your detailed answer here",
    "citations": [
        {{
            "document_name": "filename.md",
            "section_header": "Section title",
            "snippet": "Relevant text snippet from the source"
        }}
    ],
    "confidence": "high/medium/low"
}}

JSON Response:"""
        
        response = self.llm.generate(prompt)
        
        # Try to parse JSON
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                citations = [Citation(**cite) for cite in result.get('citations', [])]
                return StructuredAnswer(
                    answer=result.get('answer', response),
                    citations=citations,
                    confidence=result.get('confidence', 'medium')
                )
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        
        # Fallback: create citations from context nodes
        citations = []
        if include_citations:
            for node in context_nodes[:max_citations]:
                citations.append(Citation(
                    document_name=node.metadata.get('document_name', 'Unknown'),
                    section_header=node.metadata.get('header_path', 'Unknown'),
                    snippet=node.get_content()[:200] + "..."
                ))
        
        return StructuredAnswer(
            answer=response,
            citations=citations,
            confidence="medium"
        )
    
    def generate_simple_answer(self, query: str, context: str) -> str:
        """
        Generate a simple text answer without structured output
        
        Args:
            query: User query
            context: Context text
            
        Returns:
            Answer text
        """
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""
        
        return self.llm.generate(prompt)


def main():
    """Test LLM serving"""
    from ..core.config import load_config
    from ..core.utils import setup_logging
    
    config = load_config()
    setup_logging(config.log_level)
    
    logger.info("Testing LLM server...")
    
    # Initialize LLM server
    llm_server = LLMServer(config)
    
    # Test generation
    test_prompt = "What is a resistor in electronics?"
    response = llm_server.generate(test_prompt)
    
    logger.info(f"Prompt: {test_prompt}")
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    main()
