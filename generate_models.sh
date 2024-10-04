#!/bin/bash

# Read each line from models.txt and run the sbert.py script with the model name
while IFS= read -r model; do
    echo "Generating embeddings for model: $model"
    python sbert.py --model "$model"
done < models.txt