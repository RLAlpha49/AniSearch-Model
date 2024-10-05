#!/bin/bash

# Check if a starting model is provided
start_model="$1"
start_processing=false

# Read each line from models.txt and run the sbert.py script with the model name
while IFS= read -r model; do
    if [ "$model" == "$start_model" ]; then
        start_processing=true
    fi

    if [ "$start_processing" = true ]; then
        echo "Generating embeddings for model: $model"
        python sbert.py --model "$model"
    fi
done < models.txt