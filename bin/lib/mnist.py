# -*- coding: utf-8 -*-
"""MNIST

  MNIST: main class exported by this module.
"""

import logging
import struct
from io import FileIO
from typing import List, Tuple

import numpy as np
from numpy import ndarray

logger = logging.getLogger(__name__)

IMAGE_NUM_ROWS = 28
IMAGE_NUM_COLS = 28
IMAGE_DIM = IMAGE_NUM_ROWS * IMAGE_NUM_COLS
NUM_CLASSES = 10


class MNIST:
    """Class to read MNIST Database

    Args:
        labels_file (str): Path to a labels file
        images_file (str): Path to an images file

    Attributes:
        _images_file (str): Path to an images file
        _labels_file (str): Path to a labels file
        images (ndarray): images. Shape: (# of samples, # of cols * # of rows) before preprocess,
            otherwise (# of samples, IMAGE_NUM_ROWS, IMAGE_NUM_COLS, 1).
        labels (ndarray): labels. Shape: (# of samples,) before preprocess,
            otherwise (# of samples, NUM_CLASSES).
        num_samples (int): # of samples
    """

    def __init__(self, images_file: str, labels_file: str) -> None:
        self._images_file: str = images_file
        self._labels_file: str = labels_file
        self.images: ndarray = ...
        self.labels: ndarray = ...
        self.num_samples: int = ...

    def read(self):
        """Reads reads MNIST Database.

        This method fills labels and images fields of this object.

        Raises:
            FileNotFoundError: If labels file or images file doesn't exist.
            OSError: If fails to read labels file or images file.
        """

        with open(self._labels_file, 'br') as labelf:
            labelf: FileIO = labelf

            magic_num: bytes = labelf.read(4)
            assert magic_num == b'\x00\x00\x08\x01'

            num_labels: int = struct.unpack('>I', labelf.read(4))[0]
            self.num_samples = num_labels

            logger.info(f'Reading {num_labels} labels from {self._labels_file}...')
            labels: List[int] = []
            for _ in range(num_labels):
                label: int = int.from_bytes(labelf.read(1), 'big')
                labels.append(label)

            self.labels = np.asarray(labels)

        with open(self._images_file, 'br') as imgf:
            imgf: FileIO = imgf

            magic_num: bytes = imgf.read(4)
            assert magic_num == b'\x00\x00\x08\x03'

            num_images: int = struct.unpack('>I', imgf.read(4))[0]
            num_rows: int = struct.unpack('>I', imgf.read(4))[0]
            num_cols: int = struct.unpack('>I', imgf.read(4))[0]
            img_dim: int = num_rows * num_cols

            assert self.num_samples == num_images
            assert num_rows == IMAGE_NUM_ROWS
            assert num_cols == IMAGE_NUM_COLS

            logger.info(f'Reading {num_images} images from {self._images_file}...')
            images: ndarray = np.empty((num_images, img_dim), int)
            for i in range(num_images):
                img: bytes = imgf.read(img_dim)
                img: List[int] = struct.unpack(f'{img_dim}B', img)
                images[i] = img

            self.images = images

    def preprocess(self):
        """Preprocesses labels and images.

        The images will be divided by 255 and reshaped to (IMAGE_NUM_ROWS, IMAGE_NUM_COLS, 1).
        The labels will be converted to one-hot vector.
        """

        self.labels = np.eye(NUM_CLASSES)[self.labels]
        self.images = np.reshape(self.images / 255, (self.num_samples, IMAGE_NUM_ROWS, IMAGE_NUM_COLS, 1))

    def get(self, idx: int) -> Tuple[ndarray, ndarray]:
        """Gets ith sample.

        Args:
            idx (int): An index to get.

        Returns:
            tuple: (image, label).
        """

        return self.images[idx], self.labels[idx]
