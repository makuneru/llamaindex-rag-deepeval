"""
Pytest configuration and fixtures for Agentic RAG evaluation tests.

This module imports from the Agentic-RAG-with-LlamaIndex repository
(https://github.com/makuneru/Agentic-RAG-with-LlamaIndex)
to test the actual production components.
"""

import pytest
import os
import sys
from pathlib import Path
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Add Agentic-RAG to Python path
AGENTIC_RAG_PATH = Path(__file__).parent.parent.parent.parent / "Agentic-RAG-with-LlamaIndex"
sys.path.insert(0, str(AGENTIC_RAG_PATH / "src"))

# Import from Agentic-RAG (the actual system we're testing)
from router_engine import get_router_query_engine
from agents import create_function_calling_agent, create_multi_document_agent
from document_tools import get_doc_tools
from config import get_openai_api_key


# ============================================================================
# Test Document Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_document_path():
    """
    Get path to a sample document for testing.
    
    Returns:
        str: Path to test document
    """
    # Check environment variable first
    test_doc_path = os.getenv("TEST_DOCUMENT_PATH")
    if test_doc_path and os.path.exists(test_doc_path):
        return test_doc_path
    
    # Try to use MetaGPT paper from Agentic-RAG repo
    metagpt_path = AGENTIC_RAG_PATH / "data" / "papers" / "metagpt.pdf"
    if metagpt_path.exists():
        return str(metagpt_path)
    
    # No document available
    pytest.skip("No test document available. Set TEST_DOCUMENT_PATH or add papers to Agentic-RAG/data/papers/")


@pytest.fixture(scope="session")
def multi_document_paths():
    """Get paths to multiple test documents."""
    papers_dir = AGENTIC_RAG_PATH / "data" / "papers"
    if not papers_dir.exists():
        pytest.skip("Papers directory not found")
    
    paper_files = list(papers_dir.glob("*.pdf"))[:3]  # Use first 3 papers
    if len(paper_files) < 2:
        pytest.skip("Need at least 2 papers for multi-document tests")
    
    return [str(p) for p in paper_files]


# ============================================================================
# Agentic-RAG Component Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def router_engine(sample_document_path):
    """
    Create a router query engine from Agentic-RAG.
    
    Tests the actual router engine implementation.
    """
    engine = get_router_query_engine(sample_document_path)
    return engine


@pytest.fixture(scope="session")
def document_tools(sample_document_path):
    """
    Create document tools from Agentic-RAG.
    
    Returns:
        tuple: (vector_tool, summary_tool)
    """
    vector_tool, summary_tool = get_doc_tools(sample_document_path, "test_doc")
    return vector_tool, summary_tool


@pytest.fixture(scope="session")
def agent(document_tools):
    """
    Create a function calling agent from Agentic-RAG.
    
    Tests the actual agent implementation.
    """
    vector_tool, summary_tool = document_tools
    agent = create_function_calling_agent([vector_tool, summary_tool], verbose=False)
    return agent


@pytest.fixture(scope="session")
def multi_document_agent(multi_document_paths):
    """
    Create a multi-document agent from Agentic-RAG.
    
    Tests the actual multi-document agent implementation.
    """
    paper_names = [Path(p).name for p in multi_document_paths]
    agent = create_multi_document_agent(
        paper_names,
        data_dir=str(AGENTIC_RAG_PATH / "data" / "papers"),
        verbose=False
    )
    return agent


# ============================================================================
# Mock Fixtures for Unit Tests
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM for unit tests (no API calls)."""
    from unittest.mock import Mock
    mock = Mock(spec=OpenAI)
    mock.complete.return_value.text = "Mock LLM response"
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for unit tests (no API calls)."""
    from unittest.mock import Mock
    mock = Mock(spec=OpenAIEmbedding)
    return mock


# ============================================================================
# Test Question Fixtures
# ============================================================================

@pytest.fixture
def test_questions():
    """
    Sample test questions for evaluation.
    
    Returns:
        List of tuples: (question, expected_type)
    """
    return [
        ("What is the main topic of this document?", "general"),
        ("Can you summarize the key points?", "summary"),
        ("What are the main findings?", "findings"),
        ("What methodology was used?", "specific"),
        ("Who are the authors?", "specific"),
    ]


@pytest.fixture
def complex_questions():
    """Complex questions that require multi-step reasoning."""
    return [
        "Compare the methodology used in this paper with standard approaches",
        "What are the key contributions and how do they relate to each other?",
        "Explain the evaluation results and their significance",
    ]


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Integration tests with real API calls (slow)")
    config.addinivalue_line("markers", "evaluation: DeepEval metric evaluation tests")
    config.addinivalue_line("markers", "slow: Tests that take more than 5 seconds")

