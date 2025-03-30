# Datasets

AniSearch Model uses a variety of anime and manga datasets to provide comprehensive search capabilities. This page details the datasets used and how they're processed.

## Anime Datasets

The system combines multiple anime datasets to ensure broad coverage of titles:

1. **MyAnimeList Dataset** (`Anime.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist)
    - Contents: ~17,500 anime entries with ratings, genres, and synopses

2. **Anime Dataset 2023** (`anime-dataset-2023.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
    - Contents: Updated anime entries with recent titles

3. **Anime Database 2022** (`Anime-2022.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/harits/anime-database-2022)
    - Contents: ~15,000 anime entries with detailed metadata

4. **Anime Dataset** (`animes.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/arnavvvvv/anime-dataset)
    - Contents: Alternative set of anime entries

5. **Anime DataSet** (`anime4500.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/souradippal/anime-dataset)
    - Contents: ~4,500 popular anime titles

6. **Anime Data** (`Anime_data.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/itsnobita/anime-details/data)
    - Contents: Detailed anime information with extended descriptions

7. **Anime2** (`anime2.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/unibahmad/anime-dataset/data)
    - Contents: Additional anime entries

8. **MAL Anime** (`mal_anime.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/crxxom/all-animes-in-mal/data)
    - Contents: Comprehensive MyAnimeList data

9. **Anime 270**
    - Source: [Hugging Face](https://huggingface.co/datasets/johnidouglas/anime_270)
    - Contents: Curated set of 270 anime entries

10. **Wykonos Anime**
    - Source: [Hugging Face](https://huggingface.co/datasets/wykonos/anime)
    - Contents: Specialized anime dataset with detailed tags

## Manga Datasets

For manga search functionality, the following datasets are used:

1. **MyAnimeList Manga Dataset** (`Manga.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist?select=manga.csv)
    - Contents: ~14,000 manga entries with ratings and synopses

2. **MyAnimeList Jikan Database** (`jikan.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist-jikan?select=manga.csv)
    - Contents: Data extracted from MyAnimeList via the Jikan API

3. **Manga, Manhwa and Manhua Dataset** (`data.csv`)
    - Source: [Kaggle](https://www.kaggle.com/datasets/victorsoeiro/manga-manhwa-and-manhua-dataset)
    - Contents: Diverse collection of Japanese manga, Korean manhwa, and Chinese manhua

## Dataset Processing

The `merge_datasets.py` script handles dataset preparation:

1. **Cleaning**:
    - Removes duplicate entries
    - Standardizes text fields
    - Filters entries without synopses

2. **Merging**:
    - Combines datasets based on unique identifiers
    - Resolves conflicting information
    - Creates unified CSV files

3. **Output**:
    - `model/merged_anime_dataset.csv`: Combined anime dataset
    - `model/merged_manga_dataset.csv`: Combined manga dataset

## Dataset Structure

The final merged datasets contain these key fields:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (typically MyAnimeList ID) |
| `title` | Primary title of the anime/manga |
| `title_english` | English title (if different) |
| `synopsis` | Plot summary/description |
| `genres` | List of genres |
| `type` | Media type (TV, Movie, OVA, Manga, etc.) |
| `score` | Average user rating |
| `popularity` | Popularity ranking |
| `episodes`/`chapters` | Number of episodes/chapters |

## Light Novels

For manga searches, you can optionally include light novels:

```bash
python src/main.py search --type manga --query "Fantasy world" --include-light-novels
```

This includes entries with type "Light Novel" in the search results, which are filtered out by default.
