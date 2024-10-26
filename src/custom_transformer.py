"""
This module defines a custom T5 Encoder model that replaces ReLU activation functions with GELU.

The CustomT5EncoderModel class extends the Transformer model from the sentence_transformers library
and modifies the activation functions in the feed-forward networks of the transformer blocks to use
GELU instead of ReLU. This modification can help improve model performance since GELU has been shown
to work well in transformer architectures.
"""

from typing import Optional, Dict
from sentence_transformers import models
from torch import nn


class CustomT5EncoderModel(models.Transformer):
    """
    Custom T5 Encoder model that replaces ReLU activation functions with GELU.

    This class extends the Transformer model from the sentence_transformers library
    and modifies the activation functions in the feed-forward networks of the
    transformer blocks to use GELU instead of ReLU. GELU (Gaussian Error Linear Unit)
    is a smoother activation function that often performs better than ReLU in
    transformer architectures.

    Attributes:
        model_name_or_path (str): Name or path of the pre-trained T5 model to load.
        model_args (Optional[Dict]): Additional arguments to pass to the T5 model
            constructor.
        max_seq_length (int): Maximum sequence length for input text. Longer
            sequences will be truncated.
        do_lower_case (bool): Whether to convert input text to lowercase before
            tokenization.
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
            model_name_or_path (str): Name or path of the pre-trained T5 model to load.
            model_args (Optional[Dict]): Additional arguments to pass to the T5 model constructor.
                Defaults to an empty dict if None.
            max_seq_length (int): Maximum sequence length for input text. Default is 256.
            do_lower_case (bool): Whether to convert input text to lowercase. Default is False.
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_args=model_args if model_args is not None else {},
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
        )
        if not model_name_or_path.startswith("toobi/anime"):
            self.modify_activation(self.auto_model)

    def modify_activation(self, model):
        """
        Replace ReLU activation with GELU in all transformer blocks of the T5 encoder.

        This method iterates through all transformer blocks in the encoder and replaces
        the ReLU activation in each feed-forward network with GELU activation.

        Args:
            model: The underlying T5 transformer model whose activations will be modified.
        """
        for _, block in enumerate(model.encoder.block):
            # Accessing the feed-forward network within each block
            ff = block.layer[1].DenseReluDense
            # Replace ReLU with GELU
            ff.act = nn.GELU()
