"""
Hallucination Evaluation Tests

Tests whether RAG responses contain hallucinations (unsupported claims) using DeepEval.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from src.rag_app import RAGApplication


class TestHallucination:
    """Test for hallucinations in RAG responses."""
    
    @pytest.mark.parametrize("input_question,expected_hallucination_threshold", [
        ("What is the main topic of this document?", 0.3),  # Lower is better for hallucination
        ("Can you summarize the key points?", 0.3),
        ("What are the main findings?", 0.3),
        ("What methodology was used?", 0.3),
        ("What are the conclusions?", 0.3),
    ])
    def test_hallucination_metric(self, rag_app, input_question, expected_hallucination_threshold):
        """
        Test that RAG responses do not contain hallucinations.
        
        Hallucination metric detects unsupported claims in the response.
        Lower scores are better (less hallucination).
        
        Args:
            rag_app: RAG application fixture
            input_question: The question to ask
            expected_hallucination_threshold: Maximum acceptable hallucination score
        """
        # Get response from RAG app
        actual_output = rag_app.query(input_question)
        
        # Get retrieval context - critical for detecting hallucinations
        retrieval_context = rag_app.get_retrieval_context(input_question)
        
        # Create test case
        test_case = LLMTestCase(
            input=input_question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Create metric with threshold (lower is better)
        # Note: HallucinationMetric uses threshold differently - it's a maximum acceptable score
        hallucination_metric = HallucinationMetric(threshold=expected_hallucination_threshold)
        
        # Assert test passes
        assert_test(test_case, [hallucination_metric])
        
        # Verify score is below threshold (lower hallucination is better)
        assert hallucination_metric.score <= expected_hallucination_threshold, \
            f"Hallucination score {hallucination_metric.score} above threshold {expected_hallucination_threshold}"
    
    def test_hallucination_detailed(self, rag_app):
        """Test hallucination detection with detailed output."""
        question = "What are the main findings in this document?"
        actual_output = rag_app.query(question)
        retrieval_context = rag_app.get_retrieval_context(question)
        
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        hallucination_metric = HallucinationMetric(threshold=0.5)
        hallucination_metric.measure(test_case)
        
        # Print detailed results
        print(f"\nHallucination Score: {hallucination_metric.score}")
        print(f"Reason: {hallucination_metric.reason}")
        print(f"Input: {question}")
        print(f"Output: {actual_output[:200]}...")
        print(f"Retrieval Context: {len(retrieval_context)} chunks")
        
        # Lower score is better for hallucination
        assert hallucination_metric.score <= 0.5
    
    def test_hallucination_with_complex_queries(self, rag_app):
        """Test hallucination detection with complex, multi-part queries."""
        question = "What is this document about and what are its main contributions?"
        actual_output = rag_app.query(question)
        retrieval_context = rag_app.get_retrieval_context(question)
        
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        hallucination_metric = HallucinationMetric(threshold=0.4)
        hallucination_metric.measure(test_case)
        
        # Lower score means less hallucination
        assert hallucination_metric.score <= 0.4, \
            f"Hallucination detected: score {hallucination_metric.score} exceeds threshold"

