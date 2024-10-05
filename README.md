# AniSearchModel

This project involves generating and analyzing Sentence-BERT (SBERT) embeddings for both anime and manga datasets. The goal is to preprocess, merge, and analyze data to find the most similar synopses using SBERT models.

## Table of Contents

- [Overview](#overview)
- [Datasets Used](#datasets-used)
- [Setup](#setup)
- [Usage](#usage)
  - [Merging Datasets](#merging-datasets)
  - [Generating Embeddings](#generating-embeddings)
  - [Testing Embeddings](#testing-embeddings)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project performs the following operations:

- Loads and preprocesses multiple anime and manga datasets.
- Merges datasets based on common identifiers.
- Generates SBERT embeddings for synopses or descriptions.
- Calculates cosine similarities to find the most similar synopses or descriptions.

## Datasets Used

The following datasets are used in this project:

### Anime Datasets

1. **Anime Dataset 2023** (`anime-dataset-2023.csv`): [Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
2. **Animes** (`animes.csv`): Contains additional anime synopses and titles.
3. **Anime 2022** (`Anime-2022.csv`): Another dataset with anime synopses and titles.
4. **Anime 4500** (`anime4500.csv`): Provides further synopses for anime titles.
5. **Wykonos Dataset**: Contains descriptions and Japanese names for anime.
6. **Anime Data** (`Anime_data.csv`): Additional dataset with descriptions.
7. **Anime2** (`anime2.csv`): Includes descriptions and Japanese names.

### Manga Datasets

1. **Manga Main** (`manga.csv`): The primary manga dataset.
2. **Jikan** (`jikan.csv`): Merged via `mal_id` and `manga_id`.
3. **Data** (`data.csv`): Merged via title.

## Setup

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Ensure all datasets are placed in their respective directories under `data/anime/` and `data/manga/`.

## Usage

### Merging Datasets

The repository already contains a the merged datasets, but if you want to merge additional datasets, edit the `merge_datasets.py` file and run:

```bash
python merge_datasets.py --type anime
python merge_datasets.py --type manga
```

### Generating Embeddings

To generate SBERT embeddings for the anime dataset, you can run the `sbert.py` script for a specific model or use the provided script to generate embeddings for all models listed in `models.txt`.

#### For a Specific Model

```bash
python sbert.py --model <model_name> --type <dataset_type>
```

Replace `<model_name>` with the desired SBERT model, e.g., `all-mpnet-base-v1`. Replace `<dataset_type>` with `anime` or `manga`.

## Generating Embeddings for All Models

You can use the provided scripts to generate embeddings for all models listed in `models.txt`. These scripts will process both anime and manga datasets for each model.

### Linux

The `generate_models.sh` script is available for Linux users. To run the script, follow these steps:

1. Make the script executable:

   ```bash
   chmod +x generate_models.sh
   ```

2. Run the script:

   ```bash
   ./generate_models.sh
   ```

3. Optionally, you can specify a starting model to resume processing from a specific point:

   ```bash
   ./generate_models.sh sentence-transformers/all-MiniLM-L6-v1
   ```

### Windows (Batch Script)

The `generate_models.bat` script is available for Windows users. You can run the batch script by either double-clicking it or executing it from the command prompt:

1. Open Command Prompt and navigate to the directory containing the script.
2. Run the script:

   ```cmd
   generate_models.bat
   ```

3. Optionally, specify a starting model:

   ```cmd
   generate_models.bat sentence-transformers/all-MiniLM-L6-v1
   ```

### Windows (PowerShell Script)

The `generate_models.ps1` script is available for Windows users who prefer PowerShell. To run the script, use the following command in PowerShell:

1. Open PowerShell and navigate to the directory containing the script.
2. Run the script:

   ```powershell
   .\generate_models.ps1
   ```

3. Optionally, specify a starting model:

   ```powershell
   .\generate_models.ps1 -StartModel "sentence-transformers/all-MiniLM-L6-v1"
   ```

### Notes

- The starting model parameter is optional. If not provided, the script will process all models from the beginning of the list.
- For PowerShell, you may need to adjust the execution policy to allow script execution. You can do this by running `Set-ExecutionPolicy RemoteSigned` in an elevated PowerShell session.

These scripts automate the process of generating embeddings for all models and datasets, making it easy to manage large-scale embedding generation tasks.

### Testing Embeddings

To test the embeddings and find the most similar synopses, use the `test.py` script. Specify the model, dataset type, and the number of top results to retrieve.

```bash
python test.py --model <model_name> --type <dataset_type> --top_n <number_of_results>
```

Replace `<model_name>` with the desired SBERT model, e.g., `all-mpnet-base-v1`. Replace `<dataset_type>` with `anime` or `manga`. Replace `<number_of_results>` with the number of top results to retrieve.

### Running the Flask Application

To run the Flask application, use the `run_server.py` script. This script automatically determines the operating system and uses the appropriate server:

- On **Linux**, it uses Gunicorn.
- On **Windows**, it uses Waitress.

Run the script with:

```bash
python run_server.py
```

The application will be accessible at `http://0.0.0.0:5000/anisearchmodel`.

## Project Structure

- **merge_datasets.py**: Merges multiple anime datasets.
- **sbert.py**: Generates SBERT embeddings for the dataset.
- **test.py**: Tests the SBERT model by finding similar synopses.
- **common.py**: Contains utility functions for loading datasets and preprocessing text.
- **data/**: Directory for storing datasets.
- **model/**: Directory for storing models and embeddings.
- **model/merged_anime_dataset.csv**: Stores the merged anime dataset.
- **model/evaluation_results.json**: Stores evaluation data and results.
- **models.txt**: List of SBERT models to be used.
- **generate_models.sh**: Script to generate embeddings for all models (Linux).
- **generate_models.bat**: Script to generate embeddings for all models (Windows Batch).
- **generate_models.ps1**: Script to generate embeddings for all models (Windows PowerShell).

## Dependencies

- Python 3.6+
- pandas
- numpy
- torch
- transformers
- sentence-transformers
- tqdm
- datasets

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
