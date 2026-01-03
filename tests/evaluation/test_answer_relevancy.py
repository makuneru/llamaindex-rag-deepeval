"""
Answer Relevancy Evaluation Tests

Tests the relevancy of Agentic-RAG responses using DeepEval.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


@pytest.mark.evaluation
class TestAnswerRelevancy:
    """Test answer relevancy using DeepEval AnswerRelevancyMetric."""
    
    @pytest.mark.parametrize("input_question,expected_relevancy", [
        ("What is the main topic of this document?", 0.7),
        ("Can you summarize the key points?", 0.7),
        ("What are the main findings?", 0.7),
        ("Who are the authors?", 0.6),
        ("What methodology was used?", 0.7),
    ])
    def test_answer_relevancy_metric(self, router_engine, input_question, expected_relevancy):
        """
        Test that Agentic-RAG responses are relevant to the input questions.
        
        Args:
            router_engine: Router engine from Agentic-RAG
            input_question: The question to ask
            expected_relevancy: Minimum relevancy score threshold
        """
        # Get response from router engine
        response = router_engine.query(input_question)
        actual_output = str(response)
        
        # Get retrieval context (source nodes)
        retrieval_context = []
        if hasattr(response, 'source_nodes'):
            retrieval_context = [node.text for node in response.source_nodes]
        else:
            retrieval_context = [actual_output]  # Fallback
        
        # Create test case
        test_case = LLMTestCase(
            input=input_question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Create metric with threshold
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=expected_relevancy)
        
        # Assert test passes
        assert_test(test_case, [answer_relevancy_metric])
        
        # Verify score is above threshold
        assert answer_relevancy_metric.score >= expected_relevancy, \
            f"Answer relevancy score {answer_relevancy_metric.score} below threshold {expected_relevancy}"
    
    def test_answer_relevancy_detailed(self, router_engine):
        """Test answer relevancy with detailed output."""
        question = "What is this document about?"
        response = router_engine.query(question)
        actual_output = str(response)
        
        retrieval_context = []
        if hasattr(response, 'source_nodes'):
            retrieval_context = [node.text for node in response.source_nodes]
        else:
            retrieval_context = [actual_output]
        
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
        answer_relevancy_metric.measure(test_case)
        
        # Print detailed results
        print(f"\nAnswer Relevancy Score: {answer_relevancy_metric.score}")
        print(f"Reason: {answer_relevancy_metric.reason}")
        print(f"Input: {question}")
        print(f"Output: {actual_output[:200]}...")
        
        assert answer_relevancy_metric.score >= 0.5

