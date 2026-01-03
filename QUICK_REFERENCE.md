# 🚀 Quick Reference - Agentic RAG Testing

## ⚡ Quick Start (2 minutes)

```bash
# 1. Setup
cd llamaindex-rag-deepeval
pip install -r requirements.txt requirements-dev.txt
export OPENAI_API_KEY=your-key-here

# 2. Run tests
pytest tests/ -v
```

---

## 📋 Common Commands

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (fast, free)
pytest tests/unit/ -v -m unit

# Integration tests (slow, costs $0.08)
pytest tests/integration/ -v -m integration

# Evaluation tests (slow, costs $0.25)
pytest tests/evaluation/ -v -m evaluation

# Specific component
pytest tests/unit/test_router_engine.py -v
pytest tests/unit/test_agents.py -v
pytest tests/integration/test_multi_document.py -v
```

### Coverage

```bash
# Generate coverage report
pytest --cov=../Agentic-RAG-with-LlamaIndex/src --cov-report=html

# View in browser
open htmlcov/index.html
```

### Debugging

```bash
# Stop at first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s

# Verbose output
pytest tests/ -vv

# Run specific test
pytest tests/unit/test_agents.py::TestAgentCreation::test_agent_created_successfully -v
```

---

## 🎯 Test Markers

```bash
# Run by marker
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m evaluation     # Evaluation tests only
pytest -m slow           # Slow tests only
pytest -m "not slow"     # Skip slow tests
```

---

## 📊 What's Tested

| Component | Tests | Location |
|-----------|-------|----------|
| Router Engine | 12 | `tests/unit/test_router_engine.py` |
| Agents | 14 | `tests/unit/test_agents.py` |
| Document Tools | 10 | `tests/unit/test_document_tools.py` |
| Multi-Document | 7 | `tests/integration/test_multi_document.py` |
| Quality Metrics | 10 | `tests/evaluation/*.py` |

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY=sk-your-key-here

# Optional (defaults to MetaGPT paper)
export TEST_DOCUMENT_PATH=/path/to/test.pdf
```

### Files

- `pytest.ini` - Pytest configuration
- `requirements.txt` - Core dependencies
- `requirements-dev.txt` - Dev tools
- `tests/conftest.py` - Shared fixtures

---

## 🐛 Troubleshooting

### "No test document available"
```bash
# Use MetaGPT paper from Agentic-RAG
ls ../Agentic-RAG-with-LlamaIndex/data/papers/metagpt.pdf

# Or set custom document
export TEST_DOCUMENT_PATH=/path/to/your/doc.pdf
```

### "Cannot import router_engine"
```bash
# Verify Agentic-RAG location
ls ../Agentic-RAG-with-LlamaIndex/src/

# Should show: router_engine.py, agents.py, document_tools.py, config.py
```

### "OPENAI_API_KEY not found"
```bash
export OPENAI_API_KEY=your-key-here
```

---

## 💰 Cost Guide

| Test Type | Count | Duration | Cost |
|-----------|-------|----------|------|
| Unit | 13 | ~10 sec | $0.00 |
| Integration | 20 | ~5 min | ~$0.08 |
| Evaluation | 10 | ~8 min | ~$0.25 |
| **Full Suite** | **43** | **~13 min** | **~$0.33** |

**Tip:** Run unit tests during development (free), integration tests before commits.

---

## 📁 Repository Structure

```
llamaindex-rag-deepeval/
├── tests/
│   ├── unit/              # Fast, mocked
│   ├── integration/       # Real API
│   ├── evaluation/        # DeepEval
│   └── conftest.py        # Fixtures
├── src/
│   └── config.py          # Helpers
├── README.md              # Full docs
├── QUICK_REFERENCE.md     # This file
└── pytest.ini             # Config
```

---

## 📚 More Info

- Full documentation: `README.md`
- Agentic-RAG repository: [github.com/makuneru/Agentic-RAG-with-LlamaIndex](https://github.com/makuneru/Agentic-RAG-with-LlamaIndex)
- Implementation summary: `/Desktop/IMPLEMENTATION_COMPLETE.md`
- Test fixtures: `tests/conftest.py`

---

**Repository**: [makuneru/Agentic-RAG-with-LlamaIndex](https://github.com/makuneru/Agentic-RAG-with-LlamaIndex)  
**Status:** ✅ Production Ready | **Tests:** 43

