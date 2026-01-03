# Agentic RAG Testing Framework

Comprehensive test suite for the [Agentic-RAG-with-LlamaIndex](https://github.com/makuneru/Agentic-RAG-with-LlamaIndex) system, featuring unit tests, integration tests, and quality evaluation with DeepEval.

## 🎯 Overview

This testing framework validates the Agentic RAG system's core components:
- **Router Engine**: Query routing between summary and vector retrieval
- **Agents**: Function calling agents with multi-step reasoning
- **Document Tools**: Vector and summary query tools
- **Multi-Document**: Cross-document querying and synthesis

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- Test documents (PDFs in Agentic-RAG repo)

### Installation

1. **Install dependencies:**
   ```bash
   cd llamaindex-rag-deepeval
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   export TEST_DOCUMENT_PATH=/path/to/test/document.pdf  # Optional
   ```

3. **Clone the Agentic-RAG repository:**
   ```bash
   cd /Users/mcayanan/Desktop
   git clone https://github.com/makuneru/Agentic-RAG-with-LlamaIndex.git
   ```
   
   The testing framework automatically imports from `../Agentic-RAG-with-LlamaIndex`.
   Ensure both repositories are in the same parent directory.

### Running Tests

#### Run All Tests
```bash
pytest tests/ -v
```

#### Run by Category

**Unit Tests (Fast, Mocked):**
```bash
pytest tests/unit/ -v -m unit
```

**Integration Tests (Real API Calls):**
```bash
pytest tests/integration/ -v -m integration
```

**Evaluation Tests (DeepEval Metrics):**
```bash
pytest tests/evaluation/ -v -m evaluation
```

#### Run Specific Components

**Router Engine Tests:**
```bash
pytest tests/unit/test_router_engine.py -v
pytest tests/integration/test_router_engine.py -v
```

**Agent Tests:**
```bash
pytest tests/unit/test_agents.py -v
```

**Multi-Document Tests:**
```bash
pytest tests/integration/test_multi_document.py -v
```

### With Coverage

```bash
pytest tests/ --cov=../Agentic-RAG-with-LlamaIndex/src --cov-report=html
open htmlcov/index.html  # View coverage report
```

## 📁 Repository Structure

```
llamaindex-rag-deepeval/
├── tests/
│   ├── unit/                          # Fast unit tests
│   │   ├── test_router_engine.py      # Router engine tests
│   │   ├── test_agents.py             # Agent tests
│   │   └── test_document_tools.py     # Document tools tests
│   │
│   ├── integration/                   # Integration tests (real API)
│   │   └── test_multi_document.py     # Multi-document agent tests
│   │
│   ├── evaluation/                    # DeepEval quality tests
│   │   ├── test_answer_relevancy.py   # Answer relevancy metrics
│   │   ├── test_faithfulness.py       # Faithfulness metrics
│   │   ├── test_hallucination.py      # Hallucination detection
│   │   └── test_comprehensive_evaluation.py  # All metrics
│   │
│   └── conftest.py                    # Shared fixtures
│
├── src/                               # Utilities (minimal)
│   └── config.py                      # Configuration helpers
│
├── requirements.txt                   # Core dependencies
├── requirements-dev.txt               # Development tools
├── pytest.ini                         # Pytest configuration
└── README.md                          # This file
```

## 🧪 Test Types

### Unit Tests (`@pytest.mark.unit`)
- **Fast**: < 1 second each
- **Mocked**: No real API calls
- **Purpose**: Test component structure and logic
- **Cost**: Free

### Integration Tests (`@pytest.mark.integration`)
- **Slow**: 5-30 seconds each
- **Real API**: Actual OpenAI calls
- **Purpose**: Test end-to-end functionality
- **Cost**: ~$0.002 per test

### Evaluation Tests (`@pytest.mark.evaluation`)
- **Slow**: 10-60 seconds each
- **DeepEval**: Quality metrics (faithfulness, hallucination, etc.)
- **Purpose**: Measure response quality
- **Cost**: ~$0.005 per test

## 📊 What Gets Tested

### Router Engine
- ✅ Engine creation and initialization
- ✅ Query routing decisions (summary vs vector)
- ✅ Multiple sequential queries
- ✅ Edge cases (empty query, very long query)

### Agents
- ✅ Agent creation with tools
- ✅ Simple question answering
- ✅ Follow-up question handling
- ✅ Complex multi-step reasoning
- ✅ Information synthesis

### Document Tools
- ✅ Tool creation and metadata
- ✅ Vector tool functionality
- ✅ Summary tool functionality
- ✅ Unique tool naming

### Multi-Document Agent
- ✅ Multi-doc agent creation
- ✅ Single document queries
- ✅ Cross-document comparison
- ✅ Information synthesis across papers
- ✅ Specific paper routing

### Quality Metrics (DeepEval)
- ✅ Answer relevancy (0.7+ threshold)
- ✅ Faithfulness (0.7+ threshold)
- ✅ Hallucination detection (< 0.4 threshold)
- ✅ Summarization quality (0.7+ threshold)

## ⚙️ Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY=sk-your-key-here

# Optional - Uses MetaGPT paper from Agentic-RAG by default
export TEST_DOCUMENT_PATH=/path/to/your/test.pdf
```

### Pytest Markers

Use markers to run specific test subsets:

```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Only evaluation tests
pytest -m evaluation

# Exclude slow tests
pytest -m "not slow"
```

## 🔧 Development

### Adding New Tests

1. **Unit tests**: Add to `tests/unit/`
2. **Integration tests**: Add to `tests/integration/`
3. **Evaluation tests**: Add to `tests/evaluation/`

Example test:

```python
import pytest

@pytest.mark.unit
def test_new_feature(router_engine):
    """Test description."""
    result = router_engine.query("test question")
    assert result is not None
```

### Available Fixtures

- `router_engine`: Router query engine from Agentic-RAG
- `agent`: Function calling agent
- `multi_document_agent`: Multi-document agent
- `document_tools`: Tuple of (vector_tool, summary_tool)
- `sample_document_path`: Path to test document
- `test_questions`: List of test questions
- `complex_questions`: Complex reasoning questions

## 📈 Test Results

### Expected Pass Rates

- **Unit Tests**: 100% (all should pass)
- **Integration Tests**: 95%+ (occasional API timeouts)
- **Evaluation Tests**: 90%+ (LLM variance)

### Performance Benchmarks

| Test Type | Duration | API Calls | Cost |
|-----------|----------|-----------|------|
| Unit Tests (13) | ~10 sec | 0 | $0.00 |
| Integration Tests (20) | ~5 min | ~40 | ~$0.08 |
| Evaluation Tests (10) | ~8 min | ~50 | ~$0.25 |
| **Full Suite (43)** | **~13 min** | **~90** | **~$0.33** |

## 🐛 Troubleshooting

### Issue: "No test document available"

**Solution:**
```bash
# Option 1: Use MetaGPT paper from Agentic-RAG
cd ../Agentic-RAG-with-LlamaIndex/data/papers
# Ensure metagpt.pdf exists

# Option 2: Set custom document
export TEST_DOCUMENT_PATH=/path/to/your/doc.pdf
```

### Issue: "Cannot import from router_engine"

**Solution:**
```bash
# Verify Agentic-RAG repo exists and is at correct path
ls -la ../Agentic-RAG-with-LlamaIndex/src/

# Should show: router_engine.py, agents.py, document_tools.py, config.py
```

### Issue: "OPENAI_API_KEY not found"

**Solution:**
```bash
export OPENAI_API_KEY=your-key-here

# Or create .env file:
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Issue: Tests fail with API rate limit

**Solution:**
```bash
# Run only unit tests (no API calls)
pytest tests/unit/ -v

# Or run integration tests with delays
pytest tests/integration/ -v --durations=0 --maxfail=1
```


View coverage:
```bash
pytest --cov=../Agentic-RAG-with-LlamaIndex/src --cov-report=html
open htmlcov/index.html
```

## 🤝 Contributing

1. Add tests for new features
2. Ensure all tests pass: `pytest tests/ -v`
3. Check coverage: `pytest --cov`
4. Update this README if needed

## 📄 License

MIT License - Same as Agentic-RAG-with-LlamaIndex

## 🔗 Related

- [Agentic-RAG-with-LlamaIndex](https://github.com/makuneru/Agentic-RAG-with-LlamaIndex) - The system being tested
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [DeepEval Documentation](https://docs.confident-ai.com/)

---

**GitHub Repository:** [makuneru/Agentic-RAG-with-LlamaIndex](https://github.com/makuneru/Agentic-RAG-with-LlamaIndex)  
**Status:** ✅ Production Ready | **Tests:** 43
