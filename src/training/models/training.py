"""
This module provides utilities for configuring and training sentence transformer models.

It includes functionality for:
1. Model initialization and configuration
2. Evaluator setup for model validation
3. Loss function selection and configuration

The module supports multiple loss functions:
- Cosine Similarity Loss
- CoSENT Loss
- AnglE Loss

And provides flexible evaluation options with configurable precision levels.

Functions:
    create_model: Initialize and configure a SentenceTransformer model
    create_evaluator: Set up an evaluator for model validation
    get_loss_function: Get the appropriate loss function based on configuration

Type Definitions:
    LossType: Union type for supported loss function instances
    LossFunctionType: Union type for loss function classes
    PrecisionType: Literal type for supported precision levels

This module is designed to work with the sentence-transformers library and supports
various training configurations for fine-tuning transformer models on similarity tasks.
"""

from typing import Dict, List, Type, Union, Literal, Optional
from sentence_transformers import SentenceTransformer, losses, InputExample, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import CosineSimilarityLoss, CoSENTLoss, AnglELoss
import torch.nn as nn

from custom_transformer import CustomT5EncoderModel  # pylint: disable=import-error

LossType = Union[CosineSimilarityLoss, CoSENTLoss, AnglELoss]
LossFunctionType = Type[Union[CosineSimilarityLoss, CoSENTLoss, AnglELoss]]
PrecisionType = Literal["float32", "int8", "uint8", "binary", "ubinary"]


def create_model(
    model_name: str,
    use_custom_transformer: bool = False,
    max_seq_length: int = 843,
    device: Optional[str] = None,
) -> SentenceTransformer:
    """
    Create a SentenceTransformer model, optionally using a custom transformer.

    Args:
        model_name (str): Name or path of the pre-trained Transformer model.
        use_custom_transformer (bool): Whether to use custom transformer with modified activations.
        max_seq_length (int): Maximum sequence length for the Transformer.
        device (Optional[str]): Device to run the model on.

    Returns:
        SentenceTransformer: The initialized SentenceTransformer model.
    """
    modules: List[nn.Module] = []

    if use_custom_transformer:
        # Instantiate the custom transformer
        custom_transformer = CustomT5EncoderModel(
            model_name_or_path=model_name,
            model_args={},
            max_seq_length=max_seq_length,
            do_lower_case=False,
        )
        modules.append(custom_transformer)
    else:
        # Default transformer
        transformer = models.Transformer(model_name, max_seq_length=max_seq_length)
        modules.append(transformer)

    # Add the Pooling layer
    pooling = models.Pooling(modules[0].get_word_embedding_dimension())
    modules.append(pooling)

    # Add the Dense layer
    dense = models.Dense(
        in_features=modules[0].get_word_embedding_dimension(),
        out_features=modules[0].get_word_embedding_dimension(),
        activation_function=nn.Identity(),  # type: ignore
        bias=False,
    )
    modules.append(dense)

    # Add the Normalize layer
    normalize = models.Normalize()
    modules.append(normalize)

    # Build the SentenceTransformer model
    model = SentenceTransformer(modules=modules, device=device)
    return model


def create_evaluator(
    val_pairs: List[InputExample],
    write_csv: bool = True,
    precision: PrecisionType = "float32",
) -> EmbeddingSimilarityEvaluator:
    """Create the evaluator for model validation."""
    val_sentences_1 = [pair.texts[0] for pair in val_pairs]
    val_sentences_2 = [pair.texts[1] for pair in val_pairs]
    val_labels = [float(pair.label) for pair in val_pairs]

    return EmbeddingSimilarityEvaluator(
        val_sentences_1,
        val_sentences_2,
        val_labels,
        main_similarity="cosine",
        write_csv=write_csv,
        precision=precision,
    )


def get_loss_function(loss_name: str, model: SentenceTransformer) -> LossType:
    """Get the appropriate loss function."""
    loss_functions: Dict[str, LossFunctionType] = {
        "cosine": losses.CosineSimilarityLoss,
        "cosent": losses.CoSENTLoss,
        "angle": losses.AnglELoss,
    }

    if loss_name not in loss_functions:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return loss_functions[loss_name](model=model)
