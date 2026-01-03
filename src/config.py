"""
Configuration and API key management for the RAG evaluation system.
"""

import os
from dotenv import load_dotenv, find_dotenv


def load_env():
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    """
    Get OpenAI API key from environment variables.
    
    Returns:
        str: OpenAI API key
        
    Raises:
        ValueError: If API key is not found
    """
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file or environment variables."
        )
    return openai_api_key

