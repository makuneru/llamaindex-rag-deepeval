"""
Summarization Evaluation Tests

Tests the quality of summarization outputs from RAG responses using DeepEval.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from src.rag_app import RAGApplication


class TestSummarization:
    """Test summarization quality of RAG responses."""
    
    @pytest.mark.parametrize("input_question,expected_summarization", [
        ("Can you summarize this document?", 0.7),
        ("What are the key points?", 0.7),
        ("Give me a brief overview.", 0.7),
        ("What is the main summary?", 0.7),
        ("Summarize the main findings.", 0.7),
    ])
    def test_summarization_metric(self, rag_app, input_question, expected_summarization):
        """
        Test that RAG responses provide good summarizations.
        
        Summarization metric evaluates the quality, coherence, and completeness
        of summary outputs.
        
        Args:
            rag_app: RAG application fixture
            input_question: The summarization question to ask
            expected_summarization: Minimum summarization quality score threshold
        """
        # Get response from RAG app
        actual_output = rag_app.query(input_question)
        
        # Get retrieval context
        retrieval_context = rag_app.get_retrieval_context(input_question)
        
        # Create test case
        test_case = LLMTestCase(
            input=input_question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Create metric with threshold
        summarization_metric = SummarizationMetric(threshold=expected_summarization)
        
        # Assert test passes
        assert_test(test_case, [summarization_metric])
        
        # Verify score is above threshold
        assert summarization_metric.score >= expected_summarization, \
            f"Summarization score {summarization_metric.score} below threshold {expected_summarization}"
    
    def test_summarization_detailed(self, rag_app):
        """Test summarization quality with detailed output."""
        question = "Can you provide a comprehensive summary of this document?"
        actual_output = rag_app.query(question)
        retrieval_context = rag_app.get_retrieval_context(question)
        
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        summarization_metric = SummarizationMetric(threshold=0.5)
        summarization_metric.measure(test_case)
        
        # Print detailed results
        print(f"\nSummarization Score: {summarization_metric.score}")
        print(f"Reason: {summarization_metric.reason}")
        print(f"Input: {question}")
        print(f"Output: {actual_output[:300]}...")
        print(f"Retrieval Context: {len(retrieval_context)} chunks")
        
        assert summarization_metric.score >= 0.5
    
    def test_summarization_coherence(self, rag_app):
        """Test that summarization outputs are coherent and well-structured."""
        question = "Summarize the main points of this document in a clear and organized way."
        actual_output = rag_app.query(question)
        retrieval_context = rag_app.get_retrieval_context(question)
        
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        summarization_metric = SummarizationMetric(threshold=0.6)
        summarization_metric.measure(test_case)
        
        # Check that output has reasonable length (not too short, not too long)
        assert len(actual_output) > 50, "Summary should have substantial content"
        assert len(actual_output) < 5000, "Summary should be concise"
        
        assert summarization_metric.score >= 0.6

