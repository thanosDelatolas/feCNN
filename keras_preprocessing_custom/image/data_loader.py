"""Utilities for real-time data augmentation on image data.
"""
import warnings

import numpy as np

from .affine_transformations import (apply_affine_transform,
                                     apply_brightness_shift,
                                     apply_channel_shift, flip_axis)
from .dataframe_iterator import DataFrameIterator
from .directory_iterator import DirectoryIterator
from .numpy_array_iterator import NumpyArrayIterator


class DataLoader(object):

    def __init__(self,
                 data_format='channels_last',
                 validation_split=0.0,
                 dtype='float32'):


        self.data_format = data_format
       
        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)
        self._validation_split = validation_split

       
    def flow_from_directory(self,
                            directory_x,
                            directory_y,
                            class_mode='eeg',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            follow_links=False,
                        ):
        """Takes the path to a directory & generates batches of augmented data.

        # Arguments
            directory_y: string, path to the target directory for the y data (output of the neural net)
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM, npy, mat 
            
            directory_x: string, path to the target directory for the x data (input of the neural net)
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM, npy, mat 
                
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the subdirectory names/structure
                under `directory`, where each subdirectory will
                be treated as a different class
                (and the order of the classes, which will map to the label
                indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", "eeg". Default: "eeg".
            batch_size: Size of the batches of data (default: 32).

        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a NumPy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a NumPy array of corresponding labels.
        """
        return DirectoryIterator(
            directory_x,
            directory_y,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            follow_links=follow_links,            
        )
