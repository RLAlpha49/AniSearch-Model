# Usage Examples

This page provides practical examples of using AniSearch Model for various scenarios.

## Basic Search Examples

### Finding Action Anime with a Specific Theme

=== "Command"

    ```bash
    python src/main.py search --type anime --query "An action anime about a protagonist who gains supernatural powers and fights evil organizations"
    ```

=== "Expected Output"

    ```
    Top 5 matches for 'An action anime about a protagonist who gains supernatural powers and fights evil organizations':
    
    1. My Hero Academia (Score: 0.87)
       ID: 31964
       Synopsis excerpt: Izuku Midoriya, a boy born without superpowers in a world where they have become commonplace, but who still dreams of becoming a superhero himself, is scouted by the world's greatest hero who shares his powers with Izuku after recognizing his potential...
    
    2. One Punch Man (Score: 0.82)
       ID: 30276
       Synopsis excerpt: The seemingly ordinary and unimpressive Saitama has a rather unique hobby: being a hero. In order to pursue his childhood dream, he trained relentlessly for three years—and lost all of his hair in the process...
    
    ...
    ```

### Finding Romance Manga

=== "Command"

    ```bash
    python src/main.py search --type manga --query "A high school romance manga where the main characters start off disliking each other"
    ```

=== "Expected Output"

    ```
    Top 5 matches for 'A high school romance manga where the main characters start off disliking each other':
    
    1. Kaguya-sama: Love is War (Score: 0.91)
       ID: 90125
       Synopsis excerpt: Considered a genius due to having the highest grades in the country, Miyuki Shirogane leads the prestigious Shuchiin Academy's student council as its president, working alongside the beautiful and wealthy vice president Kaguya Shinomiya. The two are often regarded as the perfect couple...
    
    ...
    ```

## Interactive Mode Examples

### Anime Search Session

=== "Session"

    ```
    $ python src/main.py search --type anime --interactive
    
    === AniSearch Model Interactive Mode ===
    Type a description to search for anime (Ctrl+C to exit):
    
    > Anime with time travel
    
    Top 5 matches for 'Anime with time travel':
    1. Steins;Gate (Score: 0.89)
    2. Erased (Score: 0.85)
    3. Re:Zero -Starting Life in Another World- (Score: 0.79)
    4. Tokyo Revengers (Score: 0.78)
    5. The Girl Who Leapt Through Time (Score: 0.77)
    
    Enter another query:
    
    > Psychological thriller with mind games
    
    Top 5 matches for 'Psychological thriller with mind games':
    1. Death Note (Score: 0.93)
    2. Code Geass (Score: 0.86)
    3. Monster (Score: 0.85)
    4. Psycho-Pass (Score: 0.82)
    5. The Promised Neverland (Score: 0.79)
    ```

## Advanced Usage Examples

### Using Different Models

=== "Basic Model (Fast)"

    ```bash
    python src/main.py search --type anime --query "Space western with bounty hunters" --model "cross-encoder/ms-marco-TinyBERT-L-2"
    ```

=== "Default Model (Balanced)"

    ```bash
    python src/main.py search --type anime --query "Space western with bounty hunters" --model "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ```

=== "Advanced Model (Accurate)"

    ```bash
    python src/main.py search --type anime --query "Space western with bounty hunters" --model "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ```

### Including Light Novels in Manga Search

=== "Without Light Novels (Default)"

    ```bash
    python src/main.py search --type manga --query "Isekai adventure with game mechanics"
    ```

=== "With Light Novels"

    ```bash
    python src/main.py search --type manga --query "Isekai adventure with game mechanics" --include-light-novels
    ```

## Custom Training Examples

### Fine-tuning on Anime Data

=== "Command"

    ```bash
    python src/main.py train --type anime --model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
        --epochs 3 --batch-size 16 --learning-rate 2e-5 --max-samples 10000
    ```

=== "Expected Output"

    ```
    Creating training dataset...
    Generated 10000 training examples
    Starting training with batch size 16 for 3 epochs...
    Epoch 1/3: 100%|██████████| 625/625 [04:12<00:00, 2.48it/s, loss=0.142]
    Evaluation: 100%|██████████| 32/32 [00:13<00:00, 2.35it/s]
    Epoch 2/3: 100%|██████████| 625/625 [04:10<00:00, 2.49it/s, loss=0.079]
    Evaluation: 100%|██████████| 32/32 [00:13<00:00, 2.37it/s]
    Epoch 3/3: 100%|██████████| 625/625 [04:11<00:00, 2.48it/s, loss=0.055]
    Evaluation: 100%|██████████| 32/32 [00:13<00:00, 2.36it/s]
    
    ==================================================
    Training completed successfully!
    Fine-tuned model saved to: model/fine-tuned/anime-model-2023-10-25-12-30-45
    To use this model for search:
      python src/main.py search --type anime --model "model/fine-tuned/anime-model-2023-10-25-12-30-45" --query "Your query"
    ==================================================
    ```

### Creating Labeled Data Without Training

=== "Command"

    ```bash
    python src/main.py train --type manga --create-labeled-data "data/labeled_manga.csv"
    ```

=== "Expected Output"

    ```
    Creating labeled dataset...
    Generated 50000 labeled examples
    Labeled data created and saved to: data/labeled_manga.csv
    ``` 
