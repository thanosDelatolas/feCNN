"""Utilities for real-time data augmentation on image data.
"""
from ast import Raise
import multiprocessing.pool
import os

import numpy as np

from .iterator import BatchFromFilesMixin, Iterator
from .utils import _list_valid_filenames_in_directory


class DirectoryIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: string, path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        follow_links: Boolean, follow symbolic links to subdirectories
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        keep_aspect_ratio: Boolean, whether to resize images to a target size
            without aspect ratio distortion. The image is cropped in the center
            with target aspect ratio before resizing.
        dtype: Dtype to use for generated arrays.
    """
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', 'eeg' ,None}

    def __new__(cls, *args, **kwargs):
        try:
            from tensorflow.keras.utils import Sequence as TFSequence
            if TFSequence not in cls.__bases__:
                cls.__bases__ = cls.__bases__ + (TFSequence,)
        except ImportError:
            pass
        return super(DirectoryIterator, cls).__new__(cls)

    def __init__(self,
                 dir_x,
                 dir_y,
                 class_mode='eeg',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 load_file_names=False,
                 n=252300,
                 follow_links=True,
                 dtype='float32'):
        self.target_size_y = (50460,)
        self.target_size_x = (67,67)
        self.load_file_names = load_file_names

        if not self.load_file_names:
            shuffle = False
       
        self.dir_x = dir_x
        self.dir_y = dir_y
       
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype

       
        # First, count the number of samples and classes.
        self.samples_y = 0

        
        self.classes_y = []
        for subdir in sorted(os.listdir(dir_y)):
            if os.path.isdir(os.path.join(dir_y, subdir)):
                self.classes_y.append(subdir)

        self.num_classes_y = len(self.classes_y)
        self.class_indices_y = dict(zip(self.classes_y, range(len(self.classes_y))))

        self.classes_x = []
        for subdir in sorted(os.listdir(dir_x)):
            if os.path.isdir(os.path.join(dir_x, subdir)):
                self.classes_x.append(subdir)
        
        self.num_classes_x = len(self.classes_x)
        self.class_indices_x = dict(zip(self.classes_x, range(len(self.classes_x))))

        if self.load_file_names:

            pool = multiprocessing.pool.ThreadPool()

            # Second, build an index of the images
            # in the different class subfolders.
            results_y = []
            self.filenames_y = []
            i = 0

            # loader for y
            for dirpath in (os.path.join(dir_y, subdir) for subdir in self.classes_y):
                results_y.append(
                    pool.apply_async(_list_valid_filenames_in_directory,
                                    (dirpath, self.white_list_formats, None,
                                    self.class_indices_y, follow_links)))
            

            # loader for x
            results_x = []
            self.filenames_x = []
            for dirpath in (os.path.join(dir_x, subdir) for subdir in self.classes_x):
                results_x.append(
                    pool.apply_async(_list_valid_filenames_in_directory,
                                        (dirpath, self.white_list_formats, None,
                                        self.class_indices_x, follow_links)))
            classes_list = []
            for res in results_y:
                classes, filenames = res.get()
                classes_list.append(classes)
                self.filenames_y += filenames

            for res in results_x:
                classes, filenames = res.get()
                self.filenames_x += filenames

            self.samples_x = len(self.filenames_x)
            self.samples_y = len(self.filenames_y)

            print('Y samples:',self.samples_y)
            print('X samples:',self.samples_x)
            
            if self.samples_x != self.samples_y:
                raise AttributeError('X and Y must have the same amount of samples.')

            self.classes = np.zeros((self.samples_y,), dtype='int32')
            for classes in classes_list:
                self.classes[i:i + len(classes)] = classes
                i += len(classes)


            if self.num_classes_y != self.num_classes_x:
                raise AttributeError('The classes of x and y must have the same len in regression.')
            print('Found %d images belonging to %d classes.' %
                (self.samples_x, self.num_classes_y))
            pool.close()
            pool.join()
        else:
            self.samples_x = n
            self.samples_y = n
        
        super(DirectoryIterator, self).__init__(self.samples_x,
                                                batch_size,
                                                shuffle,
                                                seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None
