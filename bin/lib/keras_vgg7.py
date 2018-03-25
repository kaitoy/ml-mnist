# -*- coding: utf-8 -*-
"""KerasVGG7Classifier

  KerasVGG7Classifier: main class exported by this module.
  build_model(): build a KerasVGG7Classifier instance.
"""

import logging
from logging import Logger
from typing import List, Tuple

from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model, load_model
from numpy import float64, ndarray
from tensorflow.python.framework.ops import Tensor  # pylint: disable=no-name-in-module

from .mnist import IMAGE_NUM_COLS, IMAGE_NUM_ROWS, MNIST, NUM_CLASSES

logger: Logger = logging.getLogger(__name__)


class KerasVGG7Classifier:
    """Class to learn and classify MNIST data with CNN in Keras.

    This class build a CNN inspired by VGG net.

    Args:
        model_path (str): path to a model file.

    Attributes:
        _model (Model): model
    """

    def __init__(self, model_path: str = None) -> None:
        if model_path:
            self._model = load_model(model_path)
            logger.info('Loaded model:')
            self._model.summary(print_fn=logger.info)
        else:
            inputs: Tensor = Input(shape=(IMAGE_NUM_ROWS, IMAGE_NUM_COLS, 1))

            x: Tensor = Conv2D(filters=8, kernel_size=(2, 2), padding='same', activation='relu')(inputs)
            x = Conv2D(filters=8, kernel_size=(2, 2), padding='same', activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = BatchNormalization()(x)

            x = Conv2D(filters=16, kernel_size=(2, 2), padding='same', activation='relu')(x)
            x = Conv2D(filters=16, kernel_size=(2, 2), padding='same', activation='relu')(x)
            x = Conv2D(filters=16, kernel_size=(2, 2), padding='same', activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = BatchNormalization()(x)

            x = Flatten()(x)
            x = Dense(units=256, activation='relu')(x)
            x = Dense(units=256, activation='relu')(x)
            predictions: Tensor = Dense(NUM_CLASSES, activation='softmax')(x)

            model: Model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            self._model: Model = model
            logger.info('Built model:')
            self._model.summary(print_fn=logger.info)

    def fit(self, mnist: MNIST, batch_size: int = 512, epochs: int = 1) -> None:
        """Fits to the given data.

        Args:
            mnist (MNIST): Training set of MNIST.
            batch_size (int): Batch size
            epochs (int): # of epochs
        """

        self._model.fit(x=mnist.images, y=mnist.labels, batch_size=batch_size, epochs=epochs)

    def predict(self, images: ndarray, num_samples: int, batch_size: int = 512) -> List[int]:
        """Predicts the class (number) for the given image.

        Args:
            images (ndarray): Images.
            num_samples (int): Number of samples.
            batch_size (int): Batch size

        Returns:
            list(int): The results.
        """

        imgs: ndarray = images.reshape((num_samples, IMAGE_NUM_ROWS, IMAGE_NUM_COLS, 1))
        return self._model.predict(x=imgs, batch_size=batch_size).argmax(axis=1)

    def evaluate(self, mnist: MNIST, batch_size: int = 512) -> Tuple[float64, float64]:
        """Evaluates the trained model with the given data.

        Args:
            mnist (MNIST): Test set of MNIST.
            batch_size (int): Batch size

        Returns:
            tuple(float64, float64): The loss and accuracy.
        """

        return tuple(self._model.evaluate(x=mnist.images, y=mnist.labels, batch_size=batch_size))

    def dump(self, path: str) -> None:
        """Dump the model.

        Args:
            path (str): Path to dump.
        """

        self._model.save(path)


def build_model() -> KerasVGG7Classifier:
    """Builds a model.

    Returns:
        KerasVGG7Classifier: a KerasVGG7Classifier instance.
    """

    return KerasVGG7Classifier()


def load(path: str) -> KerasVGG7Classifier:
    """Load a model

    Args:
        path (str): Path to a model file.

    Returns:
        KerasVGG7Classifier: a KerasVGG7Classifier instance.
    """

    return KerasVGG7Classifier(path)
