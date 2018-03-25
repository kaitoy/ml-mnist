# -*- coding: utf-8 -*-
"""Module to load and use a model.

  DigitClassifier: main class exported by this module.
"""

import logging
from importlib import import_module
from typing import Any, List, Tuple

from numpy import float64

from .mnist import MNIST

logger = logging.getLogger(__name__)


class DigitClassifier:
    """Class to load and use a model.

    Args:
        model_name (str): Model name

    Attributes:
        model_name (str): Model name
        module (Any): Module
        model (Any): Model
    """

    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        self.module: Any = import_module("." + model_name, package='lib')
        self.model: Any = ...

    def build_model(self):
        """Build a model.
        """

        self.model: Any = self.module.build_model()

    def fit(self, mnist: MNIST, batch_size: int = 512, epochs: int = 1) -> None:
        """Fits to the given data.

        Args:
            mnist (MNIST): Training set of MNIST.
            batch_size (int): Batch size
            epochs (int): # of epochs
        """

        self.model.fit(mnist, batch_size=batch_size, epochs=epochs)

    def predict(self, mnist: MNIST, batch_size: int = 512) -> List[int]:
        """Predicts the class (number) for the given data.

        Args:
            mnist (MNIST): Test set of MNIST.
            batch_size (int): Batch size

        Returns:
            list(int): The results.
        """

        return self.model.predict(mnist.images, mnist.num_samples, batch_size=batch_size)

    def evaluate(self, mnist: MNIST, batch_size: int = 512) -> Tuple[float64, float64]:
        """Evaluates the trained model with the given data.

        Args:
            mnist (MNIST): Test set of MNIST.
            batch_size (int): Batch size

        Returns:
            tuple(float64, float64): The loss and accuracy.
        """

        return self.model.evaluate(mnist, batch_size=batch_size)

    def dump(self, path: str) -> None:
        """Dump this classifier

        Args:
            path (str): Path to dump.
        """

        self.model.dump(path)

    def load(self, path: str) -> None:
        """Load a model

        Args:
            path (str): Path to a model file.
        """

        self.model = self.module.load(path)
