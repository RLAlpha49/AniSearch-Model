# AniSearchModel

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue.svg)
![GitHub Workflow Status](https://github.com/RLAlpha49/AniSearchModel/actions/workflows/codeql.yml/badge.svg)
![GitHub Workflow Status](https://github.com/RLAlpha49/AniSearchModel/actions/workflows/ruff.yml/badge.svg)
![GitHub Workflow Status](https://github.com/RLAlpha49/AniSearchModel/actions/workflows/docs.yml/badge.svg)
AniSearchModel leverages Sentence-BERT (SBERT) models to generate embeddings for anime and manga synopses, enabling the calculation of semantic similarities between descriptions. This project facilitates the preprocessing, merging, and analysis of various anime and manga datasets to identify the most similar synopses.

## Table of Contents

- [Overview](#overview)
- [Datasets Used](#datasets-used)
- [Setup](#setup)
- [Usage](#usage)
  - [Merging Datasets](#merging-datasets)
  - [Generating Embeddings](#generating-embeddings)
    - [For a Specific Model](#for-a-specific-model)
    - [Generating Embeddings for All Models](#generating-embeddings-for-all-models)
  - [Testing Embeddings](#testing-embeddings)
  - [Running the Flask Application](#running-the-flask-application)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

AniSearchModel performs the following operations:

- **Data Loading and Preprocessing**: Loads multiple anime and manga datasets, cleans synopses, consolidates titles, and removes duplicates.
- **Data Merging**: Merges datasets based on common identifiers to create unified anime and manga datasets.
- **Embedding Generation**: Utilizes SBERT models to generate embeddings for synopses, facilitating semantic similarity calculations.
- **Similarity Analysis**: Calculates cosine similarities between embeddings to identify the most similar synopses or descriptions.
- **API Integration**: Provides a Flask-based API to interact with the model and retrieve similarity results.
- **Testing**: Implements a comprehensive test suite using `pytest` to ensure the reliability and correctness of all components.

## Datasets Used

### Anime Datasets

1. **MyAnimeList Dataset** (`Anime.csv`): [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist)
2. **Anime Dataset 2023** (`anime-dataset-2023.csv`): [Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
3. **Anime Database 2022** (`Anime-2022.csv`): [Kaggle](https://www.kaggle.com/datasets/harits/anime-database-2022)
4. **Anime Dataset** (`animes.csv`): [Kaggle](https://www.kaggle.com/datasets/arnavvvvv/anime-dataset)
5. **Anime DataSet** (`anime4500.csv`): [Kaggle](https://www.kaggle.com/datasets/souradippal/anime-dataset)
6. **Anime Data** (`anime_data.csv`): [Kaggle](https://www.kaggle.com/datasets/itsnobita/anime-details/data)
7. **Anime2** (`anime2.csv`): [Kaggle](https://www.kaggle.com/datasets/unibahmad/anime-dataset/data)
8. **MAL Anime** (`mal_anime.csv`): [Kaggle](https://www.kaggle.com/datasets/crxxom/all-animes-in-mal/data)
9. **Anime 270**: [Hugging Face](https://huggingface.co/datasets/johnidouglas/anime_270)
10. **Wykonos Anime**: [Hugging Face](https://huggingface.co/datasets/wykonos/anime)

### Manga Datasets

1. **MyAnimeList Manga Dataset** (`Manga.csv`): [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist?select=manga.csv)
2. **MyAnimeList Jikan Database** (`jikan.csv`): [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist-jikan?select=manga.csv)
3. **Manga, Manhwa and Manhua Dataset** (`data.csv`): [Kaggle](https://www.kaggle.com/datasets/victorsoeiro/manga-manhwa-and-manhua-dataset)

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/RLAlpha49/AniSearchModel.git
   cd AniSearchModel
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

3. **Ensure `setuptools` is installed**:

   Before running the setup script, make sure `setuptools` is installed in your virtual environment. This is typically included with Python, but you can update it with:

   ```bash
   pip install --upgrade setuptools
   ```

4. **Install the package and dependencies**:

   Use the `setup.py` script to install the package along with its dependencies. This will also handle the installation of PyTorch with CUDA support:

   ```bash
   python setup.py install
   ```

   This command will:
   - Install all required Python packages listed in `install_requires`.
   - Execute the `PostInstallCommand` to install PyTorch with CUDA support.

5. **Verify the installation**:

   After installation, you can verify that PyTorch is using CUDA by running:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   This should print `True` if CUDA is available and correctly configured.

## Usage

### Merging Datasets

The repository already contains the merged datasets, but if you want to merge additional datasets, edit the `merge_datasets.py` file and run:

```bash
python merge_datasets.py --type anime
python merge_datasets.py --type manga
```

### Generating Embeddings

To generate SBERT embeddings for the anime and manga datasets, you can use the provided scripts.

#### For a Specific Model

```bash
python sbert.py --model <model_name> --type <dataset_type>
```

Replace `<model_name>` with the desired SBERT model, e.g., `all-mpnet-base-v1`. Replace `<dataset_type>` with `anime` or `manga`.

#### Generating Embeddings for All Models

You can use the provided scripts to generate embeddings for all models listed in `models.txt`.

##### Linux

The `generate_models.sh` script is available for Linux users. To run the script, follow these steps:

1. Make the script executable:

   ```bash
   chmod +x generate_models.sh
   ```

2. Run the script:

   ```bash
   ./scripts/generate_models.sh
   ```

3. Optionally, specify a starting model:

   ```bash
   ./scripts/generate_models.sh sentence-transformers/all-MiniLM-L6-v1
   ```

### Windows (Batch Script)

1. Open Command Prompt and navigate to the directory containing the script.
2. Run the script:

   ```cmd
   scripts\generate_models.bat
   ```

3. Optionally, specify a starting model:

   ```cmd
   scripts\generate_models.bat sentence-transformers/all-MiniLM-L6-v1
   ```

### Windows (PowerShell Script)

1. Open PowerShell and navigate to the directory containing the script.
2. Run the script:

   ```powershell
   .\scripts\generate_models.ps1
   ```

3. Optionally, specify a starting model:

   ```powershell
   .\scripts\generate_models.ps1 -StartModel "sentence-transformers/all-MiniLM-L6-v1"
   ```

### Notes

- The starting model parameter is optional. If not provided, the script will process all models from the beginning of the list.
- For PowerShell, you may need to adjust the execution policy to allow script execution. You can do this by running `Set-ExecutionPolicy RemoteSigned` in an elevated PowerShell session.

### Testing Embeddings

## Testing

To ensure the reliability and correctness of the project, a comprehensive suite of tests has been implemented using `pytest`. The tests cover various components of the project, including:

### Unit Tests

- **`tests/test_model.py`**:
  - **Purpose**: Tests the functionality of model loading, similarity calculations, and evaluation result saving.
  - **Key Functions Tested**:
    - `test_anime_model`: Verifies that the anime model loads correctly, calculates similarities, and saves evaluation results as expected.
    - `test_manga_model`: Similar to `test_anime_model` but for the manga dataset.

- **`tests/test_merge_datasets.py`**:
  - **Purpose**: Validates the data preprocessing and merging functions, ensuring that names are correctly processed, synopses are cleaned, titles are consolidated, and duplicates are removed or handled appropriately.
  - **Key Functions Tested**:
    - `test_preprocess_name`: Ensures that names are preprocessed correctly by converting them to lowercase and stripping whitespace.
    - `test_clean_synopsis`: Checks that unwanted phrases are removed from synopses.
    - `test_consolidate_titles`: Verifies that multiple title columns are consolidated into a single 'title' column.
    - `test_remove_duplicate_infos`: Confirms that duplicate synopses are handled correctly.
    - `test_add_additional_info`: Tests the addition of additional synopsis information to the merged DataFrame.

- **`tests/test_sbert.py`**:
  - **Purpose**: Checks the SBERT embedding generation process, verifying that embeddings are correctly created and saved for both anime and manga datasets.
  - **Key Functions Tested**:
    - `run_sbert_command_and_verify`: Runs the SBERT command-line script and verifies that embeddings and evaluation results are generated as expected.
    - Parameterized tests for different dataset types (`anime`, `manga`) and their corresponding expected embedding files.

### API Tests

- **`tests/test_api.py`**:
  - **Purpose**: Tests the Flask API endpoints, ensuring that the `/anisearchmodel/manga` endpoint behaves as expected with valid inputs, handles missing fields gracefully, and correctly responds to internal server errors.
  - **Key Functions Tested**:
    - `test_get_manga_similarities_success`: Verifies successful retrieval of similarities with valid inputs.
    - `test_get_manga_similarities_missing_model`: Checks the API's response when the model name is missing.
    - `test_get_manga_similarities_missing_description`: Ensures appropriate handling when the description is missing.
    - Tests for internal server errors by simulating exceptions during processing.

### Test Configuration

- **`tests/conftest.py`**:
  - **Purpose**: Configures `pytest` options and fixtures, including command-line options for specifying the model name during tests.
  - **Key Features**:
    - Adds a command-line option `--model-name` to specify the model used in tests.
    - Provides a fixture `model_name` that retrieves the model name from the command-line options.

### Running the Tests

To run all the tests, navigate to the project's root directory and execute:

```bash
pytest
```

### Running Specific Tests

You can run specific tests or test modules. For example, to run only the API tests:

```bash
pytest tests/test_api.py
```

To run tests for a specific model, use:

```bash
pytest tests/test_sbert.py --model-name <model_name>
```

Replace `<model_name>` with the name of the model you want to test.

### Note

- `--model-name` can be used when running all tests or specific tests.

### Running the Flask Application

To run the Flask application, use the `run_server.py` script. This script automatically determines the operating system and uses the appropriate server. You can also specify whether to use CUDA or CPU for processing:

- On **Linux**, it uses Gunicorn.
- On **Windows**, it uses Waitress.

Run the script with:

```bash
python src/run_server.py [cuda|cpu]
```

Replace `[cuda|cpu]` with your desired device. If no device is specified, it defaults to `cpu`.

The application will be accessible at `http://0.0.0.0:5000/anisearchmodel`.

## Project Structure

This includes files and directories generated by the project which are not part of the source code.

```text
AniSearchModel
├── .github
│   └── workflows
│       ├── codeql.yml
│       └── ruff.yml
├── data
│   ├── anime
│   │   ├── Anime_data.csv
│   │   ├── Anime-2022.csv
│   │   ├── anime-dataset-2023.csv
│   │   ├── anime.csv
│   │   ├── Anime2.csv
│   │   ├── anime4500.csv
│   │   ├── animes.csv
│   │   └── mal_anime.csv
│   └── manga
│       ├── data.csv
│       ├── jikan.csv
│       └── manga.csv
├── logs
│   └── <filename>.log.<#>
├── models
│   ├── anime
│   │   └── <model_name>
│   │       ├── embeddings_Synopsis_anime_270_Dataset.npy
│   │       ├── embeddings_Synopsis_Anime_data_Dataset.npy
│   │       ├── embeddings_Synopsis_anime_dataset_2023.npy
│   │       ├── embeddings_Synopsis_Anime-2022_Dataset.npy
│   │       ├── embeddings_Synopsis_anime2_Dataset.npy
│   │       ├── embeddings_Synopsis_anime4500_Dataset.npy
│   │       ├── embeddings_Synopsis_animes_dataset.npy
│   │       ├── embeddings_Synopsis_mal_anime_Dataset.npy
│   │       ├── embeddings_Synopsis_wykonos_Dataset.npy
│   │       └── embeddings_synopsis.npy
│   ├── manga
│   │   └── <model_name>
│   │       ├── embeddings_Synopsis_data_Dataset.npy
│   │       ├── embeddings_Synopsis_jikan_Dataset.npy
│   │       └── embeddings_synopsis.npy
│   ├── evaluation_results_anime.json
│   ├── evaluation_results_manga.json
│   ├── evaluation_results.json
│   ├── merged_anime_dataset.csv
│   └── merged_manga_dataset.csv
├── scripts
│   ├── generate_models.bat
│   ├── generate_models.ps1
│   └── generate_models.sh
├── src
│   ├── __init__.py
│   ├── api.py
│   ├── common.py
│   ├── merge_datasets.py
│   ├── run_server.py
│   ├── sbert.py
│   └── test.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_merge_datasets.py
│   ├── test_model.py
│   └── test_sbert.py
├── .gitignore
├── architecture.txt
├── datasets.txt
├── LICENSE
├── models.txt
├── pytest.ini
├── README.md
├── requirements.txt
└── setup.py
```

## Dependencies

- **Python 3.6+**
- **Python Packages**:
  - pandas
  - numpy
  - torch
  - transformers
  - sentence-transformers
  - tqdm
  - datasets
  - flask
  - flask-limiter
  - waitress
  - gunicorn
  - pytest
  - pytest-order

Install all dependencies using:

```bash
python setup.py install
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
