# Enhanced RAG Pipeline Recommendation Framework

A comprehensive framework for recommending optimal RAG (Retrieval-Augmented Generation) pipeline configurations based on detailed business requirements and technical constraints.

## Features

- Comprehensive business context analysis
- Domain-specific optimizations
- Flexible deployment patterns
- Risk assessment and mitigation strategies
- Detailed implementation roadmaps
- Evolution planning and scaling guidance

## Installation

```bash
pip install -e .
```

## Project Structure

```
rag_recommender/
├── api/                # FastAPI application and routes
├── core/              # Core business logic and models
│   ├── models/        # Data models and enums
│   ├── composer/      # Pipeline composition logic
│   ├── knowledge/     # Knowledge base components
│   └── engine/        # Recommendation engine
├── templates/         # HTML templates and static files
└── tests/            # Unit and integration tests
```

## Usage

1. Start the API server:
```bash
python -m rag_recommender.api.main
```

2. Access the comprehensive questionnaire:
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

3. Use the API endpoints:
```python
import requests

response = requests.post(
    "http://localhost:8000/recommend-comprehensive",
    json={
        "primary_data_type": "pdf",
        "document_complexity": "complex",
        "content_domain": "legal",
        "current_volume": "large",
        # ... additional parameters ...
    }
)

recommendations = response.json()
```

## Testing

```bash
pytest
```

## Contributing

Please see CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE.md for details.
