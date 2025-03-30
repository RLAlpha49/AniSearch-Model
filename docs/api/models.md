# Models API

This page documents the search model components of AniSearch Model.

## Overview

The models package contains the classes responsible for loading datasets, initializing cross-encoder models, and performing semantic search operations.

The package is structured as follows:

- `BaseSearchModel`: Abstract base class providing common functionality
- `AnimeSearchModel`: Implementation for anime search
- `MangaSearchModel`: Implementation for manga search (with optional light novel support)

## Model Search Workflow

The search process works by comparing a user query against all entries in the dataset:

```mermaid
flowchart TD
    A[User Query] --> B[Search Model]
    B --> C[Generate Query Variations]
    C --> D[Batch Process Queries]
    E[(Dataset)] --> D
    D --> F[Calculate Relevance Scores]
    F --> G[Sort Results]
    G --> H[Return Top-K Results]
    
    style A fill:#e1f5fe,stroke:#0288d1
    style E fill:#fff3e0,stroke:#ff9800
    style H fill:#e8f5e9,stroke:#4caf50
```

This process ensures efficient and accurate retrieval of relevant content based on semantic understanding rather than simple keyword matching.

## BaseSearchModel

The foundation class with core functionality:

::: src.models.base_search_model.BaseSearchModel
    options:
      show_root_heading: true
      show_source: true

## AnimeSearchModel

Specialized model for anime search:

::: src.models.anime_search_model.AnimeSearchModel
    options:
      show_root_heading: true
      show_source: true

## MangaSearchModel

Specialized model for manga search:

::: src.models.manga_search_model.MangaSearchModel
    options:
      show_root_heading: true
      show_source: true
