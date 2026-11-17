# Document Question-Answering Chatbot (RAG)

This project is a Django-based retrieval-augmented generation (RAG) application for answering questions from uploaded documents. Users can upload PDFs, DOCX files, and TXT documents, ask grounded questions, and receive source-backed answers with citations.

## Features

- User registration and login
- Document upload and document library
- PDF/DOCX/TXT text extraction with page numbers
- Chunking and embedding pipeline
- Vector similarity retrieval for top-k context chunks
- LLM answer generation with citations
- Chat history and conversation flows
- Evaluation page for test questions
- Celery task structure for asynchronous ingestion
- Docker support for easy deployment

## Tech stack

- Django 5
- SQLite for local development
- Celery + Redis
- SentenceTransformers + FAISS
- OpenAI / Anthropic API support via environment variables
- PyMuPDF and python-docx for extraction

## Local setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy the environment file and add your API keys:

   ```bash
   cp .env.example .env
   ```

3. Apply migrations and create a superuser:

   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

4. Start the app:

   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```

   If port 8000 is already in use, run:

   ```bash
   python manage.py runserver 0.0.0.0:8001
   ```

5. Open the app in your browser at `http://localhost:8000/` (or `http://localhost:8001/` if you used the alternate port).

## Celery

For async processing:

```bash
celery -A rag_chatbot worker -l info
```

## Project structure

- `accounts/` – login and registration
- `documents/` – upload, extraction, chunking, indexing
- `chat/` – conversation and question answering
- `evaluation/` – evaluation flows for retrieval/answer scoring
- `templates/` – Django templates for the UI
- `static/` – CSS and JS assets

## Project documentation

This repository includes a long-term documentation set for future maintainers:

- [docs/PROJECT_ARTIFACTS_INDEX.md](docs/PROJECT_ARTIFACTS_INDEX.md)
- [docs/ARCHITECTURE_AND_CODEFLOW.md](docs/ARCHITECTURE_AND_CODEFLOW.md)
- [docs/EXPLAINED_MODULES.md](docs/EXPLAINED_MODULES.md)
- [docs/SETUP_AND_OPERATIONS.md](docs/SETUP_AND_OPERATIONS.md)
- [docs/TROUBLESHOOTING_AND_FUTURE_NOTES.md](docs/TROUBLESHOOTING_AND_FUTURE_NOTES.md)

## Notes

The app is designed to be easy to extend. If no LLM API key is set, the app falls back to a safe explanation so the project can still run locally without external credentials.

## Docker

A basic Dockerfile and docker-compose setup are included for containerized deployment.

```bash
docker-compose up --build
```
