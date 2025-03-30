"""
Dataset classes for training cross-encoder models for anime/manga search.
"""

from typing import List

from torch.utils.data import Dataset
from sentence_transformers import InputExample


class InputExampleDataset(Dataset):
    """Dataset wrapper for a list of InputExamples."""

    def __init__(self, examples: List[InputExample]):
        """
        Initialize the dataset with examples.

        Args:
            examples: List of InputExample objects
        """
        self.examples = examples

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        """
        Get an example by index.

        Args:
            idx: Index of the example to retrieve

        Returns:
            InputExample object at the specified index
        """
        return self.examples[idx]
