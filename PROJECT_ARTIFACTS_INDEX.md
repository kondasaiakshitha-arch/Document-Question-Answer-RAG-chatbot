# Project Artifacts Index

This folder is a long-term knowledge base for the Document Question Answering RAG chatbot.
It is written to explain the project in a way that remains understandable even years later.

## Purpose of this repository

This application lets a user:

- upload documents such as PDF, DOCX, and TXT files
- extract text from those files
- split large text into manageable chunks
- convert chunks into embeddings and store them
- retrieve the most relevant content for a specific question
- generate an answer grounded in the uploaded documents
- show citations with document and page references

The project uses Django as the web framework and a retrieval-augmented generation (RAG) pipeline to answer questions from local documents.

## Important files

- `manage.py` – Django entry point for running commands
- `rag_chatbot/settings.py` – project settings, environment variables, app registration
- `rag_chatbot/urls.py` – URL routing for the whole project
- `documents/models.py` – storage for uploaded documents and their text chunks
- `documents/services.py` – extraction, indexing, retrieval, reranking, and answer generation
- `chat/views.py` – user question workflow and conversation handling
- `chat/models.py` – conversation and message database models
- `evaluation/management/commands/*.py` – debug, rebuild, and evaluation commands

## Artifact files in this folder

- `ARCHITECTURE_AND_CODEFLOW.md` – architecture and request flow
- `EXPLAINED_MODULES.md` – module-by-module understanding of the codebase
- `SETUP_AND_OPERATIONS.md` – local setup, commands, migrations, and troubleshooting
- `TROUBLESHOOTING_AND_FUTURE_NOTES.md` – common problems and extension ideas

## Mental model for the project

Think of the project as three major layers:

1. Input layer
   - user uploads files
   - system stores them in the database and disk

2. Retrieval layer
   - document text is extracted
   - it is split into chunks
   - chunks are embedded
   - similarity search finds relevant passages

3. Generation layer
   - the relevant text is given to an LLM
   - the LLM answers strictly from the document context
   - citations are attached to source pages

## Why this project matters

This repository demonstrates a practical RAG pipeline without needing a complex microservice system. It is a good example of:

- Django application structure
- document ingestion and processing
- embedding workflows
- hybrid retrieval logic
- grounded answer generation with citations
- evaluation/debug tooling for improving retrieval quality

## Start here

If you are new to the project, read in this order:

1. `ARCHITECTURE_AND_CODEFLOW.md`
2. `EXPLAINED_MODULES.md`
3. `SETUP_AND_OPERATIONS.md`
4. `TROUBLESHOOTING_AND_FUTURE_NOTES.md`

This order explains the system before the operating details.
