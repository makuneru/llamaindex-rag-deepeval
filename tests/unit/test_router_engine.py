"""
Unit Tests for Agentic-RAG Router Engine

Tests the query routing logic between summary and vector retrieval.
"""

import pytest
from pathlib import Path


@pytest.mark.unit
class TestRouterEngineCreation:
    """Test router engine initialization."""
    
    def test_router_engine_created_successfully(self, router_engine):
        """Test that router engine can be created."""
        assert router_engine is not None
        assert hasattr(router_engine, 'query')
    
    def test_router_has_query_method(self, router_engine):
        """Test that router has query method."""
        assert callable(getattr(router_engine, 'query', None))


@pytest.mark.integration
class TestRouterQueryRouting:
    """Test router query routing decisions."""
    
    def test_router_handles_summary_question(self, router_engine):
        """Test router with holistic summary question."""
        question = "What is the main topic of this document?"
        response = router_engine.query(question)
        
        assert response is not None
        assert len(str(response)) > 10
        print(f"\nSummary Question Response: {str(response)[:200]}...")
    
    def test_router_handles_specific_question(self, router_engine):
        """Test router with specific detail question."""
        question = "What specific methodology was mentioned?"
        response = router_engine.query(question)
        
        assert response is not None
        assert len(str(response)) > 10
        print(f"\nSpecific Question Response: {str(response)[:200]}...")
    
    def test_router_handles_multiple_queries(self, router_engine):
        """Test router with multiple sequential queries."""
        questions = [
            "What is this about?",
            "What are the key findings?",
            "What methodology was used?"
        ]
        
        responses = []
        for question in questions:
            response = router_engine.query(question)
            responses.append(str(response))
        
        assert len(responses) == 3
        assert all(len(r) > 10 for r in responses)
        print(f"\nProcessed {len(responses)} queries successfully")


@pytest.mark.integration
@pytest.mark.slow
class TestRouterEdgeCases:
    """Test router edge cases."""
    
    def test_router_handles_empty_query(self, router_engine):
        """Test router behavior with empty string."""
        response = router_engine.query("")
        # Should handle gracefully
        assert response is not None
    
    def test_router_handles_very_long_query(self, router_engine):
        """Test router with very long query."""
        long_query = "What is " * 50 + "the main topic?"
        response = router_engine.query(long_query)
        
        assert response is not None
        assert len(str(response)) > 0

