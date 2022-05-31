# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
import numpy as np
from typing import Union, Iterable  # noqa


def extract_random_patches(
    images,
    patch_size,
    no_patches,
    flatten=True,
):
    # type: (np.ndarray, Union[int, Iterable[int]], int, bool) -> np.ndarray
    """Extract patches from batch of images at randomly sampled positions

    :param images: Image batch, is (no_images, height, width, no_channels)
    :param patch_size: Patch height and width (identical value if one input is provided)
    :param no_patches: Number of patches to extract
    :param flatten: Whether to flatten height, width, channels dimensions
    :return: Extracted image patches, is (no_patches, height, width, no_channels) if flatten=True,
     else (no_patches, height*width*no_channels)

    Adapted based on implementation by Georgios Exarchakis
    """
    assert (
        np.ndim(images) == 4
    ), "Input must have dimensions (no_images, height, width, no_channels)"
    no_images, height, width, no_channels = images.shape
    patch_height, patch_width = (
        patch_size if isinstance(patch_size, Iterable) else (patch_size, patch_size)
    )
    data = np.zeros((no_patches, patch_height, patch_width, no_channels), dtype="float64")
    indi = np.random.randint(0, no_images, no_patches)
    indh = np.random.randint(0, height - patch_height, no_patches)
    indw = np.random.randint(0, width - patch_width, no_patches)
    for i, ind in enumerate(indi):
        data[i] = images[ind, indh[i] : indh[i] + patch_height, indw[i] : indw[i] + patch_width, :]
    return data.reshape(no_patches, -1) if flatten else data

def extract_random_3Dpatches(
    image,
    patch_height,
    patch_width,
    patch_length,
    no_patches,
    flatten=True,
):
    # type: (np.ndarray, Union[int, Iterable[int]], int, bool) -> np.ndarray
    """Extract patches from 3D block image at randomly sampled positions

    :param images: Image batch, is (height, width, length)
    :param patch_size: Patch height and width (identical value if one input is provided)
    :param no_patches: Number of patches to extract
    :param flatten: Whether to flatten height, width, channels dimensions
    :return: Extracted image patches, is (no_patches, height, width, no_channels) if flatten=True,
     else (no_patches, height*width*no_channels)
    """
    assert (
        np.ndim(image) == 3
    ), "Input must have dimensions (height, width, length)"
    height, width, length = image.shape

    data = np.zeros((no_patches, patch_height, patch_width, patch_length), dtype="float64")
    indh = np.random.randint(0, height - patch_height, no_patches)
    indw = np.random.randint(0, width - patch_width, no_patches)
    indd = np.random.randint(0, length - patch_length, no_patches)
    for i in range(no_patches):
        data[i] = image[indh[i] : indh[i] + patch_height, indw[i] : indw[i] + patch_width, indd[i] : indd[i] + patch_length]
    data = data.reshape(no_patches, -1) if flatten else data
    return to.from_numpy(data).to(dtype=to.float32)
