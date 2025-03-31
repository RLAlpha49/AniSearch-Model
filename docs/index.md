# AniSearch Model

A semantic search engine that matches natural language descriptions with anime and manga titles using cross-encoder transformer models.

## Overview

AniSearch Model leverages transformer-based cross-encoder models to enable semantic search capabilities for anime and manga content. Instead of relying on keyword matching, the system uses natural language understanding to find relevant titles based on user descriptions.

This project implements a sophisticated search system that allows users to find anime or manga that match their descriptions. Instead of keyword matching, it uses semantic understanding to identify relevant content.

## Key Features

- **Semantic Search**: Find anime/manga by describing what you're looking for in natural language
- **Cross-Encoder Models**: Uses state-of-the-art transformer models for accurate matching
- **Support for Both Anime and Manga**: Specialized models for each content type
- **Interactive Mode**: Continuous search functionality for exploration
- **Fine-tuning Support**: Train custom models on anime/manga data

## Quick Start

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: For documentation building
pip install -r requirements-docs.txt

# Optional: For development and linting
pip install -r requirements-dev.txt

# Download and prepare datasets
python src/merge_datasets.py

# Search for anime
python src/main.py search --type anime --query "An adventure about pirates searching for treasure"

# Search for manga
python src/main.py search --type manga --query "A story about a boy who becomes a hero"

# Interactive search mode
python src/main.py search --type anime --interactive
```
