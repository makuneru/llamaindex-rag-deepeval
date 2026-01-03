"""
Unit Tests for Agentic-RAG Document Tools

Tests the document querying tools (vector and summary).
"""

import pytest


@pytest.mark.unit
class TestDocumentToolsCreation:
    """Test document tools initialization."""
    
    def test_document_tools_created(self, document_tools):
        """Test that document tools are created successfully."""
        vector_tool, summary_tool = document_tools
        
        assert vector_tool is not None
        assert summary_tool is not None
    
    def test_vector_tool_has_metadata(self, document_tools):
        """Test that vector tool has proper metadata."""
        vector_tool, _ = document_tools
        
        assert hasattr(vector_tool, 'metadata')
        assert 'name' in vector_tool.metadata or hasattr(vector_tool.metadata, 'name')
    
    def test_summary_tool_has_metadata(self, document_tools):
        """Test that summary tool has proper metadata."""
        _, summary_tool = document_tools
        
        assert hasattr(summary_tool, 'metadata')
        assert 'name' in summary_tool.metadata or hasattr(summary_tool.metadata, 'name')


@pytest.mark.integration
class TestVectorTool:
    """Test vector tool functionality."""
    
    def test_vector_tool_callable(self, document_tools):
        """Test that vector tool can be called."""
        vector_tool, _ = document_tools
        
        # Tools should be callable
        assert callable(vector_tool) or hasattr(vector_tool, 'query_engine')
    
    def test_vector_tool_specific_query(self, document_tools):
        """Test vector tool with specific query."""
        vector_tool, _ = document_tools
        
        # Vector tool should handle specific queries
        # Note: Testing through agent is more appropriate
        # This is just structure verification
        assert vector_tool is not None


@pytest.mark.integration
class TestSummaryTool:
    """Test summary tool functionality."""
    
    def test_summary_tool_callable(self, document_tools):
        """Test that summary tool can be called."""
        _, summary_tool = document_tools
        
        # Tools should be callable
        assert callable(summary_tool) or hasattr(summary_tool, 'query_engine')
    
    def test_summary_tool_holistic_query(self, document_tools):
        """Test summary tool with holistic query."""
        _, summary_tool = document_tools
        
        # Summary tool should handle holistic queries
        # Note: Testing through agent is more appropriate
        # This is just structure verification
        assert summary_tool is not None


@pytest.mark.unit
class TestToolConfiguration:
    """Test tool configuration and naming."""
    
    def test_tools_have_unique_names(self, document_tools):
        """Test that vector and summary tools have unique names."""
        vector_tool, summary_tool = document_tools
        
        # Get tool names
        vector_name = getattr(vector_tool.metadata, 'name', None) or vector_tool.metadata.get('name', '')
        summary_name = getattr(summary_tool.metadata, 'name', None) or summary_tool.metadata.get('name', '')
        
        assert vector_name != summary_name
        assert 'vector' in vector_name.lower() or 'tool' in vector_name.lower()
        assert 'summary' in summary_name.lower() or 'tool' in summary_name.lower()

