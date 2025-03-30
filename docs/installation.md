# Installation

This guide walks you through setting up AniSearch Model on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Setup Steps

### 1. Get the Code

Either clone the repository using Git:

```bash
git clone https://github.com/RLAlpha49/AniSearch-Model.git
cd AniSearch-Model
```

Or download and extract the ZIP file from the [GitHub repository](https://github.com/RLAlpha49/AniSearch-Model).

### 2. Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

This will install:

- Sentence-transformers
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- tqdm
- and other dependencies

### 3. Prepare the Datasets

Download and process the anime and manga datasets:

```bash
python src/merge_datasets.py
```

This script will:

1. Download datasets from various sources
2. Clean and process the data
3. Create merged datasets in the `model/` directory:
   - `merged_anime_dataset.csv`
   - `merged_manga_dataset.csv`

!!! note
    The dataset preparation step may take several minutes depending on your internet connection and system performance.

## Verifying Installation

To verify that everything works correctly, try running a simple search:

```bash
python src/main.py search --type anime --query "School comedy with romance" --results 3
```

If you see search results displayed, your installation is successful!

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you're using Python 3.8 or higher:

    ```bash
    python --version
    ```

2. Ensure pip is up to date:

    ```bash
    pip install --upgrade pip
    ```

3. For GPU acceleration (optional), check your PyTorch installation has CUDA support:

    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

4. If you face issues with specific packages, try installing them individually with specific versions as listed in `requirements.txt`.
