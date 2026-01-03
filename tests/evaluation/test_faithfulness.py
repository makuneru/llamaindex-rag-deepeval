"""
Faithfulness Evaluation Tests

Tests whether RAG responses are faithful to the source documents using DeepEval.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from src.rag_app import RAGApplication


class TestFaithfulness:
    """Test faithfulness of RAG responses to source documents."""
    
    @pytest.mark.parametrize("input_question,expected_faithfulness", [
        ("What is the main topic of this document?", 0.7),
        ("Can you summarize the key points?", 0.7),
        ("What are the main findings?", 0.7),
        ("What methodology was used?", 0.7),
        ("What are the conclusions?", 0.7),
    ])
    def test_faithfulness_metric(self, rag_app, input_question, expected_faithfulness):
        """
        Test that RAG responses are faithful to the source documents.
        
        Faithfulness checks if the response is supported by the retrieval context.
        
        Args:
            rag_app: RAG application fixture
            input_question: The question to ask
            expected_faithfulness: Minimum faithfulness score threshold
        """
        # Get response from RAG app
        actual_output = rag_app.query(input_question)
        
        # Get retrieval context - critical for faithfulness evaluation
        retrieval_context = rag_app.get_retrieval_context(input_question)
        
        # Create test case
        test_case = LLMTestCase(
            input=input_question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Create metric with threshold
        faithfulness_metric = FaithfulnessMetric(threshold=expected_faithfulness)
        
        # Assert test passes
        assert_test(test_case, [faithfulness_metric])
        
        # Verify score is above threshold
        assert faithfulness_metric.score >= expected_faithfulness, \
            f"Faithfulness score {faithfulness_metric.score} below threshold {expected_faithfulness}"
    
    def test_faithfulness_detailed(self, rag_app):
        """Test faithfulness with detailed output and reasoning."""
        question = "What are the main findings in this document?"
        actual_output = rag_app.query(question)
        retrieval_context = rag_app.get_retrieval_context(question)
        
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        faithfulness_metric = FaithfulnessMetric(threshold=0.5)
        faithfulness_metric.measure(test_case)
        
        # Print detailed results
        print(f"\nFaithfulness Score: {faithfulness_metric.score}")
        print(f"Reason: {faithfulness_metric.reason}")
        print(f"Input: {question}")
        print(f"Output: {actual_output[:200]}...")
        print(f"Retrieval Context: {len(retrieval_context)} chunks")
        
        assert faithfulness_metric.score >= 0.5
    
    def test_faithfulness_with_multiple_context_chunks(self, rag_app):
        """Test faithfulness evaluation with multiple retrieval context chunks."""
        question = "What is this document about?"
        actual_output = rag_app.query(question)
        retrieval_context = rag_app.get_retrieval_context(question)
        
        # Ensure we have multiple context chunks
        assert len(retrieval_context) > 0, "Should have at least one context chunk"
        
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        faithfulness_metric = FaithfulnessMetric(threshold=0.6)
        faithfulness_metric.measure(test_case)
        
        assert faithfulness_metric.score >= 0.6

