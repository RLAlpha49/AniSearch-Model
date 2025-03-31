"""
# Training Datasets

Dataset implementations for training and fine-tuning cross-encoder models for anime/manga search.

This module provides specialized dataset classes that are compatible with PyTorch's
data loading utilities and optimized for training cross-encoder models. The datasets
handle the conversion between input examples (containing queries, documents, and labels)
and the formats required by the SentenceTransformers framework.

## Features

- Compatible with PyTorch's Dataset interface and DataLoader
- Support for InputExample format from SentenceTransformers
- Efficient memory usage through lazy loading of examples
- Type-safe implementation with proper annotations

## Usage Context

These dataset classes are primarily used in:

1. Model fine-tuning processes
2. Training pipelines for cross-encoder models
3. Evaluation workflows for measuring model performance

The datasets serve as the bridge between raw data and the training process,
ensuring that examples are properly formatted and accessible to the model.
"""

from typing import List

from sentence_transformers import InputExample
from torch.utils.data import Dataset


class InputExampleDataset(Dataset):
    """
    PyTorch Dataset wrapper for a collection of SentenceTransformers InputExamples.

    This dataset class adapts a list of InputExample objects (from SentenceTransformers)
    to be compatible with PyTorch's data loading utilities. It enables efficient
    batch processing during training and evaluation of cross-encoder models.

    InputExamples typically contain:
    - A pair of texts (query and document)
    - A label indicating relevance or similarity
    - An optional identifier

    This dataset implementation allows seamless integration with PyTorch's DataLoader
    for efficient batching, shuffling, and parallel data loading during training.

    Attributes:
        examples: A list of InputExample objects containing the text pairs and labels
                 for training or evaluation.

    Example:
        ```python
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader

        # Create example data
        examples = [
            InputExample(texts=['anime query', 'anime description'], label=1.0),
            InputExample(texts=['unrelated query', 'anime description'], label=0.0),
            # ... more examples
        ]

        # Create dataset
        dataset = InputExampleDataset(examples)

        # Create DataLoader for training
        train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Use in training loop
        for batch in train_dataloader:
            # Process batch...
            pass
        ```
    """

    def __init__(self, examples: List[InputExample]):
        """
        Initialize the dataset with a list of InputExample objects.

        Args:
            examples: List of InputExample objects from SentenceTransformers.
                Each example should contain a pair of texts and a label.
                For cross-encoder training, each example typically contains:
                - texts[0]: The query text
                - texts[1]: The document text
                - label: A float value indicating relevance (typically 0 to 1)
        """
        self.examples = examples

    def __len__(self) -> int:
        """
        Return the number of examples in the dataset.

        This method is required by PyTorch's Dataset interface and is called by
        DataLoader to determine the size of the dataset and the number of batches.

        Returns:
            int: The total number of examples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        """
        Retrieve an example by its index.

        This method is required by PyTorch's Dataset interface and is called by
        DataLoader during batch generation. It retrieves a single example by its
        index in the examples list.

        Args:
            idx: Integer index of the example to retrieve, must be in range
                0 <= idx < len(self).

        Returns:
            InputExample: The example at the specified index, containing text pairs
                and a label.

        Raises:
            IndexError: If idx is out of bounds for the examples list.
        """
        return self.examples[idx]
