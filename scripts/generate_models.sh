#!/bin/bash

# Set default start model if not provided
start_model="${1:-sentence-transformers/all-distilroberta-v1}"
start_processing=false

# Define dataset types
dataset_types=("anime" "manga")

# Read each line from models.txt and run the sbert.py script with the model name
while IFS= read -r model; do
    if [ "$model" == "$start_model" ]; then
        start_processing=true
    fi

    if [ "$start_processing" = true ]; then
        for dataset_type in "${dataset_types[@]}"; do
            echo "Generating embeddings for model: $model on dataset: $dataset_type"
            python sbert.py --model "$model" --type "$dataset_type"
        done
    fi
done < models.txt