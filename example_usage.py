"""
Example Usage Script

Demonstrates how to use the LlamaIndex RAG with DeepEval framework.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_app import RAGApplication
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    SummarizationMetric
)
from deepeval.test_case import LLMTestCase


def main():
    """Main example function."""
    
    # Check for document path
    document_path = os.getenv("TEST_DOCUMENT_PATH")
    if not document_path:
        print("Please set TEST_DOCUMENT_PATH environment variable")
        print("Example: export TEST_DOCUMENT_PATH=/path/to/document.pdf")
        return
    
    if not os.path.exists(document_path):
        print(f"Document not found: {document_path}")
        return
    
    print(f"Initializing RAG application with document: {document_path}")
    
    # Initialize RAG application
    app = RAGApplication(
        document_path=document_path,
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    print("RAG application initialized successfully!\n")
    
    # Example 1: Simple Query
    print("=" * 60)
    print("Example 1: Simple Query")
    print("=" * 60)
    question = "What is this document about?"
    response = app.query(question)
    print(f"Question: {question}")
    print(f"Response: {response[:200]}...\n")
    
    # Example 2: Answer Relevancy Evaluation
    print("=" * 60)
    print("Example 2: Answer Relevancy Evaluation")
    print("=" * 60)
    question = "What are the main findings?"
    actual_output = app.query(question)
    retrieval_context = app.get_retrieval_context(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.6)
    answer_relevancy_metric.measure(test_case)
    
    print(f"Question: {question}")
    print(f"Answer Relevancy Score: {answer_relevancy_metric.score:.3f}")
    print(f"Reason: {answer_relevancy_metric.reason}")
    print(f"Response: {actual_output[:200]}...\n")
    
    # Example 3: Faithfulness Evaluation
    print("=" * 60)
    print("Example 3: Faithfulness Evaluation")
    print("=" * 60)
    question = "What methodology was used?"
    actual_output = app.query(question)
    retrieval_context = app.get_retrieval_context(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    faithfulness_metric = FaithfulnessMetric(threshold=0.6)
    faithfulness_metric.measure(test_case)
    
    print(f"Question: {question}")
    print(f"Faithfulness Score: {faithfulness_metric.score:.3f}")
    print(f"Reason: {faithfulness_metric.reason}")
    print(f"Response: {actual_output[:200]}...\n")
    
    # Example 4: Hallucination Detection
    print("=" * 60)
    print("Example 4: Hallucination Detection")
    print("=" * 60)
    question = "What are the conclusions?"
    actual_output = app.query(question)
    retrieval_context = app.get_retrieval_context(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    hallucination_metric = HallucinationMetric(threshold=0.4)
    hallucination_metric.measure(test_case)
    
    print(f"Question: {question}")
    print(f"Hallucination Score: {hallucination_metric.score:.3f} (lower is better)")
    print(f"Reason: {hallucination_metric.reason}")
    print(f"Response: {actual_output[:200]}...\n")
    
    # Example 5: Summarization Quality
    print("=" * 60)
    print("Example 5: Summarization Quality")
    print("=" * 60)
    question = "Can you summarize the key points?"
    actual_output = app.query(question)
    retrieval_context = app.get_retrieval_context(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    summarization_metric = SummarizationMetric(threshold=0.6)
    summarization_metric.measure(test_case)
    
    print(f"Question: {question}")
    print(f"Summarization Score: {summarization_metric.score:.3f}")
    print(f"Reason: {summarization_metric.reason}")
    print(f"Response: {actual_output[:200]}...\n")
    
    # Example 6: Comprehensive Evaluation
    print("=" * 60)
    print("Example 6: Comprehensive Evaluation (All Metrics)")
    print("=" * 60)
    question = "What is this document about and what are its main contributions?"
    actual_output = app.query(question)
    retrieval_context = app.get_retrieval_context(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    metrics = [
        AnswerRelevancyMetric(threshold=0.6),
        FaithfulnessMetric(threshold=0.6),
        HallucinationMetric(threshold=0.4),
        SummarizationMetric(threshold=0.6)
    ]
    
    print(f"Question: {question}")
    print("\nEvaluating with all metrics...\n")
    
    for metric in metrics:
        metric.measure(test_case)
        metric_name = metric.__class__.__name__.replace("Metric", "")
        print(f"{metric_name}: {metric.score:.3f}")
        if hasattr(metric, 'reason') and metric.reason:
            print(f"  Reason: {metric.reason[:100]}...")
    
    print(f"\nResponse: {actual_output[:300]}...")
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

