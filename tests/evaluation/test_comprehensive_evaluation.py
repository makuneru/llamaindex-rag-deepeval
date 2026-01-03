"""
Comprehensive Evaluation Tests

Tests all metrics together for a complete evaluation of RAG system quality.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    SummarizationMetric
)
from deepeval.test_case import LLMTestCase
from src.rag_app import RAGApplication


class TestComprehensiveEvaluation:
    """Comprehensive evaluation using all metrics together."""
    
    @pytest.mark.parametrize("input_question", [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the main findings?",
        "What methodology was used?",
    ])
    def test_all_metrics_together(self, rag_app, input_question):
        """
        Test RAG response quality using all evaluation metrics simultaneously.
        
        This provides a comprehensive view of response quality across multiple dimensions:
        - Answer Relevancy: Is the answer relevant to the question?
        - Faithfulness: Is the answer supported by the source documents?
        - Hallucination: Does the answer contain unsupported claims?
        - Summarization: Is the summary quality good?
        
        Args:
            rag_app: RAG application fixture
            input_question: The question to evaluate
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
        
        # Create all metrics
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.6)
        faithfulness_metric = FaithfulnessMetric(threshold=0.6)
        hallucination_metric = HallucinationMetric(threshold=0.4)  # Lower is better
        summarization_metric = SummarizationMetric(threshold=0.6)
        
        # Run all metrics
        metrics = [
            answer_relevancy_metric,
            faithfulness_metric,
            hallucination_metric,
            summarization_metric
        ]
        
        # Assert all tests pass
        assert_test(test_case, metrics)
        
        # Print comprehensive results
        print(f"\n{'='*60}")
        print(f"Comprehensive Evaluation Results")
        print(f"{'='*60}")
        print(f"Input: {input_question}")
        print(f"\nAnswer Relevancy Score: {answer_relevancy_metric.score}")
        print(f"Faithfulness Score: {faithfulness_metric.score}")
        print(f"Hallucination Score: {hallucination_metric.score} (lower is better)")
        print(f"Summarization Score: {summarization_metric.score}")
        print(f"\nOutput: {actual_output[:200]}...")
        print(f"{'='*60}\n")
        
        # Verify all metrics meet thresholds
        assert answer_relevancy_metric.score >= 0.6
        assert faithfulness_metric.score >= 0.6
        assert hallucination_metric.score <= 0.4
        assert summarization_metric.score >= 0.6
    
    def test_evaluation_dataset(self, rag_app):
        """
        Test evaluation using multiple questions as a dataset.
        
        This simulates evaluating a RAG system on a test dataset.
        """
        test_questions = [
            "What is this document about?",
            "What are the main findings?",
            "Can you summarize the key points?",
            "What methodology was used?",
        ]
        
        results = []
        
        for question in test_questions:
            actual_output = rag_app.query(question)
            retrieval_context = rag_app.get_retrieval_context(question)
            
            test_case = LLMTestCase(
                input=question,
                actual_output=actual_output,
                retrieval_context=retrieval_context
            )
            
            # Use moderate thresholds for dataset evaluation
            answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
            faithfulness_metric = FaithfulnessMetric(threshold=0.5)
            
            answer_relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)
            
            results.append({
                "question": question,
                "answer_relevancy": answer_relevancy_metric.score,
                "faithfulness": faithfulness_metric.score,
            })
        
        # Print dataset results
        print(f"\n{'='*60}")
        print(f"Dataset Evaluation Results ({len(test_questions)} questions)")
        print(f"{'='*60}")
        for i, result in enumerate(results, 1):
            print(f"\nQuestion {i}: {result['question']}")
            print(f"  Answer Relevancy: {result['answer_relevancy']:.3f}")
            print(f"  Faithfulness: {result['faithfulness']:.3f}")
        print(f"{'='*60}\n")
        
        # Verify average scores meet thresholds
        avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        
        assert avg_relevancy >= 0.5, f"Average relevancy {avg_relevancy} below threshold"
        assert avg_faithfulness >= 0.5, f"Average faithfulness {avg_faithfulness} below threshold"

