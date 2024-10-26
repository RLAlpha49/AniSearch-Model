"""
Module implementing early stopping functionality for model training.

This module provides an early stopping callback that monitors a metric during training
and stops the training process if no improvement is seen for a specified number of
evaluations. This helps prevent overfitting by stopping training when the model
performance plateaus or starts to degrade on validation data.
"""

import logging


class EarlyStoppingCallback:
    """
    Callback to implement early stopping during training.

    Attributes:
        patience (int): Number of evaluations with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        best_score (float): Best score observed so far.
        counter (int): Number of consecutive evaluations with no improvement.
        stop_training (bool): Flag to indicate whether to stop training.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.stop_training = False

    def on_evaluate(self, score: float, epoch: int, steps: int):  # pylint: disable=unused-argument
        """
        Evaluate current score and update early stopping state.

        Args:
            score (float): Current evaluation score to compare against best score
            epoch (int): Current training epoch number
            steps (int): Current training step number

        This method compares the current score against the best score seen so far.
        If the score does not improve by at least min_delta for patience number
        of evaluations, it sets stop_training to True.
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logging.info("EarlyStoppingCounter: %d/%d", self.counter, self.patience)
            if self.counter >= self.patience:
                logging.info("Early stopping triggered.")
                self.stop_training = True
        else:
            self.best_score = score
            self.counter = 0
