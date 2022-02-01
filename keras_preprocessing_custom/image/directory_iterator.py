"""Utilities for real-time data augmentation on image data.
"""
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
                 directory_x,
                 directory_y,
                 classes=None,
                 class_mode='eeg',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 
                 follow_links=False,
                 dtype='float32'):
        self.target_size = (50460,)
        # super(DirectoryIterator, self).set_processing_attrs(image_data_generator,
        #                                                     target_size,
        #                                                     color_mode,
        #                                                     data_format,
        #                                                     save_to_dir,
        #                                                     save_prefix,
        #                                                     save_format,
        #                                                     subset,
        #                                                     interpolation,
        #                                                     keep_aspect_ratio)
        self.directory_x = directory_x
        self.directory_y = directory_y
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory_y)):
                if os.path.isdir(os.path.join(directory_y, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results_y = []
        self.filenames_y = []
        i = 0

        # loader for y
        for dirpath in (os.path.join(directory_y, subdir) for subdir in classes):
            results_y.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.white_list_formats, None,
                                  self.class_indices, follow_links)))
        
        # loader for x
        results_x = []
        self.filenames_x = []
        for dirpath in (os.path.join(directory_y, subdir) for subdir in classes):
            results_x.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                    (dirpath, self.white_list_formats, None,
                                    self.class_indices, follow_links)))
        classes_list = []
        for res in results_y:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames_y += filenames

        for res in results_x:
            _, filenames = res.get()
            self.filenames_x += filenames

        self.samples = len(self.filenames_y)
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))
        pool.close()
        pool.join()
        
        super(DirectoryIterator, self).__init__(self.samples,
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