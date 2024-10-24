"""
This module defines a custom Transformer model that replaces ReLU activation functions with GELU.

The CustomT5EncoderModel class extends the Transformer model from the sentence_transformers library
and modifies the activation functions in the feed-forward networks of the transformer blocks to use
GELU instead of ReLU.
"""

from typing import Optional, Dict
from sentence_transformers import models
from torch import nn


class CustomT5EncoderModel(models.Transformer):
    """
    Custom Transformer model that replaces activation functions with GELU.

    This class extends the Transformer model from the sentence_transformers library
    and modifies the activation functions in the feed-forward networks of the transformer
    blocks to use GELU instead of ReLU.

    Attributes:
        model_name_or_path (str): Name or path of the pre-trained Transformer model.
        model_args (Optional[Dict]): Additional arguments for the Transformer model.
        max_seq_length (int): Maximum sequence length.
        do_lower_case (bool): Whether to convert the input to lowercase.
    """
    def __init__(
        self,
        model_name_or_path: str,
        model_args: Optional[Dict] = None,
        max_seq_length: int = 256,
        do_lower_case: bool = False,
    ):
        """
        Initialize the CustomT5EncoderModel.

        Args:
            model_name_or_path (str): Name or path of the pre-trained Transformer model.
            model_args (Optional[Dict]): Additional arguments for the Transformer model.
            max_seq_length (int): Maximum sequence length.
            do_lower_case (bool): Whether to convert the input to lowercase.
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_args=model_args if model_args is not None else {},
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
        )
        self.modify_activation(self.auto_model)

    def modify_activation(self, model):
        """
        Replace ReLU activation with GELU in all transformer blocks.

        Args:
            model: The underlying Transformer model.
        """
        for _, block in enumerate(model.encoder.block):
            # Accessing the feed-forward network within each block
            ff = block.layer[1].DenseReluDense
            # Replace ReLU with GELU
            ff.act = nn.GELU()
