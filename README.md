# AniSearch Model

A semantic search engine that matches natural language descriptions with anime and manga titles using cross-encoder transformer models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Search for Anime](#search-for-anime)
  - [Search for Manga](#search-for-manga)
  - [List Available Models](#list-available-models)
  - [API Server](#api-server)
- [Models](#models)
  - [Pre-trained Models](#pre-trained-models)
  - [Fine-tuned Models](#fine-tuned-models)
- [Project Structure](#project-structure)
- [Datasets Used](#datasets-used)
  - [Anime Datasets](#anime-datasets)
  - [Manga Datasets](#manga-datasets)
- [Training Custom Models](#training-custom-models)
  - [Training Parameters](#training-parameters)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a cross-encoder-based search system that allows users to find anime or manga that match their descriptions. Instead of keyword matching, it uses semantic understanding to identify relevant content.

## Features

- **Semantic Search**: Find anime/manga by describing what you're looking for in natural language
- **Cross-Encoder Models**: Uses state-of-the-art transformer models for accurate matching
- **Support for Both Anime and Manga**: Specialized models for each content type
- **Interactive Mode**: Continuous search functionality for exploration
- **Fine-tuning Support**: Train custom models on anime/manga data
- **API Server**: FastAPI-based REST API with multi-worker support for high concurrency

## Installation

### Prerequisites

- Python 3.8+
- pip
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/RLAlpha49/AniSearch-Model.git
    cd anime-search-model
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    > **Note for GPU Acceleration**: If you want to use your NVIDIA GPU for faster processing, install PyTorch with CUDA support:
    >
    > ```bash
    > # For CUDA 12.6
    > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    >
    > # For other CUDA versions, visit: https://pytorch.org/get-started/locally/
    > ```
    >
    > After installation, you can verify CUDA is available with:
    >
    > ```python
    > import torch
    > print(f"CUDA available: {torch.cuda.is_available()}")
    > ```

3. Download and prepare the datasets:

    ```bash
    python src/merge_datasets.py
    ```

## Usage

### Search for Anime

```bash
# Search for anime with a description
python src/main.py search --type anime --query "An adventure about pirates searching for treasure"

# Interactive search mode
python src/main.py search --type anime --interactive

# Specify a different model
python src/main.py search --type anime --query "A story about giant humanoid robots" --model "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

### Search for Manga

```bash
# Search for manga with a description
python src/main.py search --type manga --query "A story about a boy who becomes a hero"

# Include light novels in search results
python src/main.py search --type manga --query "Fantasy adventure with game elements" --include-light-novels
```

### List Available Models

```bash
# List pre-trained models
python src/main.py search --list-models

# List both pre-trained and fine-tuned models
python src/main.py search --list-fine-tuned
```

### API Server

The project includes a FastAPI-based REST API server that exposes the search functionality through HTTP endpoints. This allows you to integrate the search capability into other applications or build a web frontend.

```bash
# Start the API server with default settings
python -m src.api

# Start the API server with custom CORS settings
python -m src.api --cors-origins="http://localhost:3000,https://yourdomain.com" --cors-methods="GET,POST"

# Configure server performance
python -m src.api --workers=4 --limit-concurrency=100 --timeout=60

# Production mode: Enable only search endpoints for security
python -m src.api --enable-routes=search

# Production mode with high concurrency (4 workers, search endpoints only)
python -m src.api --enable-routes=search --workers=4
```

Available configuration options:

- `--host`: Host to bind the server to (default: "0.0.0.0")
- `--port`: Port to bind the server to (default: 8000)
- `--cors-origins`: Comma-separated list of allowed origins for CORS (default: "*")
- `--cors-methods`: Comma-separated list of allowed HTTP methods for CORS (default: "*")
- `--cors-headers`: Comma-separated list of allowed HTTP headers for CORS (default: "*")
- `--workers`: Number of worker processes (default: half of CPU cores)
- `--limit-concurrency`: Maximum number of concurrent connections (default: 50)
- `--timeout`: Timeout for keep-alive connections in seconds (default: 30)
- `--enable-routes`: Comma-separated list of routes to enable (options: "all", "search", "health", "models"; default: "all")

The server will start at <http://localhost:8000> and automatically use multiple workers based on your CPU cores for handling concurrent requests. You can access the interactive API documentation at <http://localhost:8000/docs>.

Example API request:

```bash
# Search for anime with GPU acceleration (if available)
curl -X POST "http://localhost:8000/search/anime" \
  -H "Content-Type: application/json" \
  -d '{"query": "A story about robots and AI"}'

# Search for manga with cpu
curl -X POST "http://localhost:8000/search/manga?device=cpu" \
  -H "Content-Type: application/json" \
  -d '{"query": "A fantasy adventure in a magical world"}'
```

The API includes built-in error handling, request validation, and performance tracking. For GPU acceleration, make sure you've installed PyTorch with CUDA support as described in the setup section.

## Models

The system supports various cross-encoder models:

### Pre-trained Models

- **MS Marco models**: Optimized for information retrieval (recommended)
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)
  - `cross-encoder/ms-marco-MiniLM-L-12-v2` (more accurate but slower)
  - `cross-encoder/ms-marco-TinyBERT-L-2` (fastest but less accurate)

It is also possible to use any cross-encoding supported model with Sentence Transformers. Plenty are available on [Hugging Face](https://huggingface.co/).

### Fine-tuned Models

You can also train your own custom models optimized for anime/manga search. Fine-tuned models are saved to `model/fine-tuned/` and can be used like pre-trained models.

## Project Structure

```text
├── data/                # Raw datasets
│   ├── anime/           # Anime datasets
│   └── manga/           # Manga datasets
├── model/               # Model files
│   ├── fine-tuned/      # Fine-tuned models
│   ├── merged_anime_dataset.csv  # Processed anime dataset
│   └── merged_manga_dataset.csv  # Processed manga dataset
├── src/                 # Source code
│   ├── cli/             # Command-line interface
│   ├── models/          # Search model implementations
│   ├── training/        # Training infrastructure
│   ├── utils/           # Utility functions
│   ├── main.py          # Entry point script
│   └── merge_datasets.py # Dataset processing
└── requirements.txt     # Project dependencies
```

## Datasets Used

### Anime Datasets

1. **MyAnimeList Dataset** (`anime.csv`): [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist)
2. **Anime Dataset 2023** (`anime-dataset-2023.csv`): [Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
3. **Anime Database 2022** (`Anime-2022.csv`): [Kaggle](https://www.kaggle.com/datasets/harits/anime-database-2022)
4. **Anime Dataset** (`animes.csv`): [Kaggle](https://www.kaggle.com/datasets/arnavvvvv/anime-dataset)
5. **Anime DataSet** (`anime4500.csv`): [Kaggle](https://www.kaggle.com/datasets/souradippal/anime-dataset)
6. **Anime Data** (`Anime_data.csv`): [Kaggle](https://www.kaggle.com/datasets/itsnobita/anime-details/data)
7. **Anime2** (`Anime2.csv`): [Kaggle](https://www.kaggle.com/datasets/unibahmad/anime-dataset/data)
8. **MAL Anime** (`mal_anime.csv`): [Kaggle](https://www.kaggle.com/datasets/crxxom/all-animes-in-mal/data)
9. **Anime 270**: [Hugging Face](https://huggingface.co/datasets/johnidouglas/anime_270)
10. **Wykonos Anime**: [Hugging Face](https://huggingface.co/datasets/wykonos/anime)

### Manga Datasets

1. **MyAnimeList Manga Dataset** (`Manga.csv`): [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist?select=manga.csv)
2. **MyAnimeList Jikan Database** (`jikan.csv`): [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist-jikan?select=manga.csv)
3. **Manga, Manhwa and Manhua Dataset** (`data.csv`): [Kaggle](https://www.kaggle.com/datasets/victorsoeiro/manga-manhwa-and-manhua-dataset)

## Training Custom Models

You can fine-tune custom models on anime/manga datasets:

```bash
# Train a model for anime
python src/main.py train --type anime --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3

# Train a model for manga (including light novels)
python src/main.py train --type manga --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3 --include-light-novels

# Create labeled data without training
python src/main.py train --type anime --create-labeled-data "data/labeled_anime.csv"
```

### Training Parameters

- `--model`: Base model to fine-tune
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate for optimizer
- `--max-samples`: Maximum number of training samples
- `--loss`: Loss function type (default: "mse")
- `--scheduler`: Learning rate scheduler (default: "linear")
- `--seed`: Random seed for reproducibility

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
