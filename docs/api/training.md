# Training API

This page documents the model training components of AniSearch Model.

## Overview

The training package provides functionality for fine-tuning cross-encoder models on anime and manga data. It includes:

- Base trainer class with common functionality
- Specialized trainers for anime and manga
- Dataset handling utilities
- Training utilities

## Data Processing Flow

The following diagram illustrates how data flows through the training process:

```mermaid
flowchart LR
    A[(Raw Dataset)] --> B[Load Dataset]
    B --> C[Filter Light Novels]
    C --> D[Clean Data]
    D --> E[Generate Training Pairs]
    E --> F[Create Query Variations]
    F --> G[Create Positive/Negative Examples]
    G --> H[Prepare Model Input]
    H --> I[Fine-tune Model]
    
    subgraph Manga Specific
        C
    end
    
    subgraph Both Trainers
        D
        E
        F
        G
        H
        I
    end
    
    style A fill:#e3f2fd,stroke:#1976d2
    style I fill:#e8f5e9,stroke:#4caf50
```

## Similarity Score Calculation

When generating synthetic training data, the system calculates similarity scores between entries based on their metadata:

```mermaid
flowchart TD
    A[Start] --> B[Parse Genre Lists]
    B --> C[Parse Theme Lists]
    C --> D{Both Lists Empty?}
    D -->|Yes| E[Return 0.0]
    D -->|No| F[Calculate Jaccard Similarity]
    F --> G[Weight Themes Higher]
    G --> H[Cap Score at 0.8]
    H --> I[Return Final Score]
    
    style A fill:#e3f2fd,stroke:#1976d2
    style E fill:#ffebee,stroke:#f44336
    style I fill:#e8f5e9,stroke:#4caf50
```

This process ensures that synthetic training pairs reflect meaningful relationships between entries based on their genres and themes.

## Base Trainer

The foundation class with core training functionality:

::: src.training.base_trainer.BaseModelTrainer
    options:
      show_root_heading: true
      show_source: true

## Anime Trainer

Specialized trainer for anime models:

::: src.training.anime_trainer.AnimeModelTrainer
    options:
      show_root_heading: true
      show_source: true

## Manga Trainer

Specialized trainer for manga models:

::: src.training.manga_trainer.MangaModelTrainer
    options:
      show_root_heading: true
      show_source: true

## Dataset

Dataset implementation for cross-encoder training:

::: src.training.dataset.InputExampleDataset
    options:
      show_root_heading: true
      show_source: true

## Training Utilities

Helper functions for model training:

::: src.training.utils
    options:
      show_root_heading: false
      show_source: true
