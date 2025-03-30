# Usage Guide

This guide covers all the ways you can use AniSearch Model for finding anime and manga content.

## Basic Commands

AniSearch Model provides two primary modes:

1. `search` - For finding anime/manga matching a description
2. `train` - For training custom models on anime/manga data

## Searching for Content

### Anime Search

Search for anime that matches your description:

```bash
python src/main.py search --type anime --query "A story about high school students with supernatural powers"
```

### Manga Search

Search for manga that matches your description:

```bash
python src/main.py search --type manga --query "A dark fantasy with powerful monsters and swordsmen"
```

Include light novels in your manga search:

```bash
python src/main.py search --type manga --query "Fantasy adventure with game elements" --include-light-novels
```

### Interactive Search Mode

For continuous searching without restarting the program:

```bash
python src/main.py search --type anime --interactive
```

In interactive mode:

- Type your search query and press Enter
- Results will be displayed
- Enter a new query or press Ctrl+C to exit

### Controlling Search Results

Adjust the number of results returned:

```bash
python src/main.py search --type anime --query "Space exploration" --results 10
```

Change batch size for processing (affects performance):

```bash
python src/main.py search --type anime --query "Romance comedy" --batch-size 32
```

### Using Different Models

List available pre-trained models:

```bash
python src/main.py search --list-models
```

List both pre-trained and your fine-tuned models:

```bash
python src/main.py search --list-fine-tuned
```

Specify a different model for search:

```bash
python src/main.py search --type anime --query "Mecha anime with war" --model "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

Use a fine-tuned model:

```bash
python src/main.py search --type anime --query "Samurai historical" --model "model/fine-tuned/anime-model-2023-10-25"
```

## Training Custom Models

### Basic Training

Train a model on anime data:

```bash
python src/main.py train --type anime --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3
```

Train a model on manga data:

```bash
python src/main.py train --type manga --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3
```

Include light novels in manga training:

```bash
python src/main.py train --type manga --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --include-light-novels
```

### Advanced Training Options

Customize training parameters:

```bash
python src/main.py train --type anime --model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --max-samples 10000 \
    --eval-steps 100
```

Create labeled data without training:

```bash
python src/main.py train --type anime --create-labeled-data "data/labeled_anime.csv"
```

Use custom labeled data for training:

```bash
python src/main.py train --type manga --model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --labeled-data "data/my_labeled_manga.csv"
```

### Available Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Base model to fine-tune | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` |
| `--epochs` | Number of training epochs | `3` |
| `--batch-size` | Training batch size | `16` |
| `--learning-rate` | Learning rate for optimizer | `2e-5` |
| `--max-samples` | Maximum number of training samples | `50000` |
| `--loss` | Loss function type (`"mse"` or `"cosine"`) | `"mse"` |
| `--scheduler` | Learning rate scheduler | `"linear"` |
| `--seed` | Random seed for reproducibility | `42` |
| `--eval-steps` | Steps between evaluations | `250` |
