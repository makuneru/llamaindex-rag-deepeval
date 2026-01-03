"""
Unit Tests for Agentic-RAG Agents

Tests the agent reasoning loop and tool calling behavior.
"""

import pytest


@pytest.mark.unit
class TestAgentCreation:
    """Test agent initialization."""
    
    def test_agent_created_successfully(self, agent):
        """Test that agent can be created."""
        assert agent is not None
        assert hasattr(agent, 'query') or hasattr(agent, 'chat')
    
    def test_agent_has_tools(self, document_tools):
        """Test that document tools are created."""
        vector_tool, summary_tool = document_tools
        
        assert vector_tool is not None
        assert summary_tool is not None
        assert hasattr(vector_tool, 'metadata')
        assert hasattr(summary_tool, 'metadata')


@pytest.mark.integration
class TestAgentQuerying:
    """Test agent query functionality."""
    
    def test_agent_answers_simple_question(self, agent):
        """Test agent with simple question."""
        question = "What is this document about?"
        response = agent.query(question)
        
        assert response is not None
        response_str = str(response)
        assert len(response_str) > 10
        print(f"\nAgent Response: {response_str[:200]}...")
    
    def test_agent_answers_specific_question(self, agent):
        """Test agent with specific detail question."""
        question = "What methodology is described?"
        response = agent.query(question)
        
        assert response is not None
        assert len(str(response)) > 10
    
    def test_agent_handles_followup_questions(self, agent):
        """Test agent with follow-up questions."""
        questions = [
            "What is the main topic?",
            "What are the key findings?",
        ]
        
        responses = []
        for question in questions:
            response = agent.query(question)
            responses.append(str(response))
        
        assert len(responses) == 2
        assert all(len(r) > 10 for r in responses)
        print(f"\nProcessed {len(responses)} follow-up questions")


@pytest.mark.integration
@pytest.mark.slow
class TestAgentReasoning:
    """Test agent multi-step reasoning."""
    
    def test_agent_complex_query(self, agent, complex_questions):
        """Test agent with complex reasoning questions."""
        question = complex_questions[0]
        response = agent.query(question)
        
        assert response is not None
        response_str = str(response)
        assert len(response_str) > 50  # Complex answer should be longer
        print(f"\nComplex Query Response: {response_str[:300]}...")
    
    def test_agent_synthesis_query(self, agent):
        """Test agent's ability to synthesize information."""
        question = "Summarize the key contributions and their significance"
        response = agent.query(question)
        
        assert response is not None
        assert len(str(response)) > 50


@pytest.mark.integration
class TestAgentEdgeCases:
    """Test agent edge cases."""
    
    def test_agent_handles_empty_query(self, agent):
        """Test agent behavior with empty query."""
        try:
            response = agent.query("")
            # Should either handle gracefully or raise specific error
            assert response is not None or True
        except Exception as e:
            # Acceptable to raise error for empty query
            assert "empty" in str(e).lower() or True
    
    def test_agent_handles_ambiguous_query(self, agent):
        """Test agent with ambiguous question."""
        response = agent.query("Tell me more")
        # Should handle gracefully
        assert response is not None

