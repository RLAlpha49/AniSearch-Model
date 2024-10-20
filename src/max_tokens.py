import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# Function to calculate max token count for a given dataset
def calculate_max_tokens(dataset_path, synopsis_columns, models, batch_size=64):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Dictionary to store the highest token count for each model
    model_max_token_counts = {}

    for model_name in models:
        # Initialize the tokenizer for the current model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=100000, clean_up_tokenization_spaces=True
        )

        # Variable to store the maximum token count for the current model
        max_tokens = 0

        for column in synopsis_columns:
            if column not in df.columns:
                print(f"Column '{column}' not found in dataset. Skipping...")
                continue

            # Drop NaN values and convert to list
            synopses = df[column].dropna().tolist()

            # Process in batches
            for i in tqdm(
                range(0, len(synopses), batch_size),
                desc=f"Processing {model_name} - {column}",
            ):
                batch = synopses[i : i + batch_size]
                # Tokenize the batch and count the tokens
                tokenized_batch = tokenizer(
                    batch, add_special_tokens=True, max_length=100000
                )
                for tokens in tokenized_batch["input_ids"]:
                    token_count = len(tokens)
                    # Update max_tokens if the current token_count is higher
                    if token_count > max_tokens:
                        max_tokens = token_count

        # Store the maximum token count for the current model
        model_max_token_counts[model_name] = max_tokens

    return model_max_token_counts


# List of models to test
models = [
    "toobi/anime",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-MiniLM-L6-v1",
    "sentence-transformers/all-MiniLM-L12-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v1",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-roberta-large-v1",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/gtr-t5-large",
    "sentence-transformers/gtr-t5-xl",
    "sentence-transformers/multi-qa-distilbert-dot-v1",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/paraphrase-distilroberta-base-v2",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "sentence-transformers/sentence-t5-base",
    "sentence-transformers/sentence-t5-large",
    "sentence-transformers/sentence-t5-xl",
    "sentence-transformers/sentence-t5-xxl",
]

# Columns to process for anime dataset
anime_synopsis_columns = [
    "synopsis",
    "Synopsis anime_dataset_2023",
    "Synopsis animes dataset",
    "Synopsis anime_270 Dataset",
    "Synopsis Anime-2022 Dataset",
    "Synopsis anime4500 Dataset",
    "Synopsis wykonos Dataset",
    "Synopsis Anime_data Dataset",
    "Synopsis anime2 Dataset",
    "Synopsis mal_anime Dataset",
]

# Columns to process for manga dataset
manga_synopsis_columns = [
    "synopsis",
    "Synopsis jikan Dataset",
    "Synopsis data Dataset",
]

# Calculate max tokens for anime dataset
anime_max_tokens = calculate_max_tokens(
    "model/merged_anime_dataset.csv", anime_synopsis_columns, models
)
print("Anime Dataset:")
for model_name, max_tokens in anime_max_tokens.items():
    print(f"Highest token count for model '{model_name}': {max_tokens}")

# Calculate max tokens for manga dataset
manga_max_tokens = calculate_max_tokens(
    "model/merged_manga_dataset.csv", manga_synopsis_columns, models
)
print("\nManga Dataset:")
for model_name, max_tokens in manga_max_tokens.items():
    print(f"Highest token count for model '{model_name}': {max_tokens}")

# Find and print the overall maximum token count for anime and manga
max_tokens_anime = max(anime_max_tokens.values())
max_tokens_manga = max(manga_max_tokens.values())

print(f"\nOverall maximum token count for Anime Dataset: {max_tokens_anime}")
print(f"Overall maximum token count for Manga Dataset: {max_tokens_manga}")
