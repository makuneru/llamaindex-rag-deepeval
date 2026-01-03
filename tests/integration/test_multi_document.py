"""
Integration Tests for Multi-Document Agent

Tests the multi-document agent's ability to query across multiple documents.
"""

import pytest


@pytest.mark.integration
class TestMultiDocumentAgentCreation:
    """Test multi-document agent initialization."""
    
    def test_multi_document_agent_created(self, multi_document_agent):
        """Test that multi-document agent can be created."""
        assert multi_document_agent is not None
        assert hasattr(multi_document_agent, 'query') or hasattr(multi_document_agent, 'chat')


@pytest.mark.integration
@pytest.mark.slow
class TestMultiDocumentQuerying:
    """Test multi-document agent querying."""
    
    def test_agent_queries_single_document(self, multi_document_agent):
        """Test multi-doc agent with single document query."""
        question = "What is the first document about?"
        response = multi_document_agent.query(question)
        
        assert response is not None
        response_str = str(response)
        assert len(response_str) > 10
        print(f"\nSingle Doc Response: {response_str[:200]}...")
    
    def test_agent_compares_documents(self, multi_document_agent):
        """Test multi-doc agent comparing documents."""
        question = "Compare the methodologies across these papers"
        response = multi_document_agent.query(question)
        
        assert response is not None
        response_str = str(response)
        assert len(response_str) > 50  # Comparison should be detailed
        print(f"\nComparison Response: {response_str[:300]}...")
    
    def test_agent_synthesizes_across_documents(self, multi_document_agent):
        """Test multi-doc agent synthesizing information."""
        question = "What are the common themes across these documents?"
        response = multi_document_agent.query(question)
        
        assert response is not None
        assert len(str(response)) > 50


@pytest.mark.integration
@pytest.mark.slow
class TestMultiDocumentEdgeCases:
    """Test multi-document agent edge cases."""
    
    def test_agent_handles_specific_paper_query(self, multi_document_agent):
        """Test querying about a specific paper."""
        # This tests if agent can route to correct document
        question = "What is the main contribution of the first paper?"
        response = multi_document_agent.query(question)
        
        assert response is not None
        assert len(str(response)) > 10
    
    def test_agent_handles_general_query(self, multi_document_agent):
        """Test general query across all documents."""
        question = "What are the key findings?"
        response = multi_document_agent.query(question)
        
        assert response is not None
        assert len(str(response)) > 10

