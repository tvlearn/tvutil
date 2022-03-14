# -*- coding: utf-8 -*-

from __future__ import division

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap  # noqa
from typing import Optional, Tuple, Dict, List  # noqa


def make_grid(
    images, nrow=None, surrounding=1, surrounding_value=None, padding=1, pad_value=None, repeat=1
):
    # type: (np.ndarray, Optional[int], int, Optional[float], int, Optional[float], int) -> np.ndarray  # noqa
    """Create image grid

    :param images: images to be merged into grid, shape must be (no_images, no_channels, height,
                           width)
    :param nrow: Number of images in horizontal direction
    :param surrounding: Surround images by line of this width in pixels
    :param surrounding_value: Value for pixels surrounding images  (set to min amplitude if not
                              specified)
    :param padding: Space between images
    :param pad_value: Value for padded pixels (set to max amplitude if not specified)
    :param repeat: Repeat each pixel of each image by this factor to yield larger figure
    :return: images as grid, is (no_channels, summed_and_padded_heights, summed_and_padded_widths)

    Converted from PyTorch to Numpy based on
    https://pytorch.org/vision/stable/_modules/torchvision/utils.html#make_grid

    For LICENSING and COPYRIGHT for this function see pytorch/vision's license at:
    https://github.com/pytorch/vision/blob/main/LICENSE
    """
    assert (
        np.ndim(images) == 4
    ), "Input must be 4-dim. image with shape (no_images, no_channels, height, width)"

    images = images.repeat(repeat, axis=2).repeat(repeat, axis=3)

    no_images, no_channels, height, width = images.shape

    nrow = no_images if nrow is None else nrow
    nrow = min(nrow, no_images)
    ncol = int(np.ceil(no_images / nrow))
    height, width = int(height + padding + 2 * surrounding), int(width + padding + 2 * surrounding)

    surrounding_value = np.min(images) if surrounding_value is None else surrounding_value
    pad_value = np.max(images) if pad_value is None else pad_value

    grid = np.ones((no_channels, height * ncol + padding, width * nrow + padding)) * pad_value

    k = 0
    for y in range(ncol):
        for x in range(nrow):
            if k >= no_images:
                break
            grid[
                :, y * height + padding : (y + 1) * height, x * width + padding : (x + 1) * width
            ] = surrounding_value
            grid[
                :,
                y * height + padding + surrounding : (y + 1) * height - surrounding,
                x * width + padding + surrounding : (x + 1) * width - surrounding,
            ] = images[k]
            k += 1
    return grid


def scale(x, new_range):
    # type: (np.ndarray, List[float]) -> np.ndarray
    """Rescale values in x to fill [new_range[0], new_range[1]]

    :param x: Array to rescale
    :param new_range: Target range, [low, high]
    :return: Array with rescaled amplitudes

    Converted from PyTorch to Numpy based on
    https://pytorch.org/vision/stable/_modules/torchvision/utils.html#make_grid

    For LICENSING and COPYRIGHT for this function see pytorch/vision's license at:
    https://github.com/pytorch/vision/blob/main/LICENSE
    """
    return (x - np.min(x)) / max(x.max() - x.min(), 1e-10) * np.diff(new_range) + new_range[0]


def _stack_with_black_white(black_value, white_value, min_value, max_value, cmap):
    # type: (float, float, float, float, LinearSegmentedColormap) -> LinearSegmentedColormap
    """Append black and white to colormap

    :param black_value: This value will be plotted in black
    :param white_value: This value will be plotted in white
    :param min_value: Minimum value in data
    :param max_value: Maximum value in data
    :param cmap: Colormap used for visualization of `values`
    :return: Stacked colormap

    Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
    Licensed under the Academic Free License version 3.0
    """
    stacked = np.vstack(
        (
            plt.cm.gray([black_value]),
            cmap(np.linspace(min_value, max_value, 254)),
            plt.cm.gray([white_value]),
        )
    )
    return mcolors.LinearSegmentedColormap.from_list("{}-bw".format(cmap.name), stacked)


def make_grid_with_black_boxes_and_white_background(
    global_clim=True, sym_clim=True, cmap=plt.cm.jet, eps=0.02, **kwargs
):
    # type: (bool, bool, LinearSegmentedColormap, float, Dict) -> Tuple[np.ndarray, LinearSegmentedColormap, float, float, str]  # noqa
    """Create image grid with black boxes around images and white figure background
    :param global_clim: If True, all images will use the same colormap with lower and upper limits
                        limits set to the normalized minimum and maximum value in the data,
                        respectively. Otherwise, each image will use a different colormap with
                        limits set to the normalized minimum and maximum value of the respective
                        image.
    :param sym_clim: If True, the lower and upper limits of the colormap will be set to plus and
                     minus the normalized maximum absolute value of the data, respectively.
                     Otherwise, the limits will be set equal to the normalized minimum and maximum
                     value of the data.
    :param cmap: Matplotlib colormap
    :param eps: Data values can fill [eps, 1. - eps]. The value 0. is used for surrounding boxes
                and plotted in black; the value 1. is used for padding areas and plotted in white.
    :param kwargs: For further arguments, see docs of `make_grid`
    :return: Tuple containing normalized images as grid with shape (no_channels,
             summed_and_padded_heights, summed_and_padded_widths), colormap, lower limit of color
             range, upper limit of color range, suffix indicating whether color scaling is
             global/local and sym./unsym.

    Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
    Licensed under the Academic Free License version 3.0
    """
    assert "images" in kwargs
    kwargs = deepcopy(kwargs)
    images = kwargs["images"]
    assert isinstance(images, np.ndarray)  # to make mypy happy
    if global_clim:
        if sym_clim:
            max_abs = np.max(np.abs(images))
            normalized_min = (
                0.0 if np.min(images) == -max_abs else (np.min(images) - (-max_abs)) / (2 * max_abs)
            )
            normalized_max = (
                1.0 if np.max(images) == max_abs else (np.max(images) - (-max_abs)) / (2 * max_abs)
            )
            images = scale(images, [normalized_min + eps, normalized_max - eps])
            scale_suff = "_global_sym"
        else:
            images = scale(images, [eps, 1.0 - eps])
            scale_suff = "_global_unsym"
    else:
        for i in range(images.shape[0]):
            if sym_clim:
                max_abs = np.max(np.abs(images[i]))
                normalized_min = (
                    0.0
                    if np.min(images[i]) == -max_abs
                    else (np.min(images[i]) - (-max_abs)) / (2 * max_abs)
                )
                normalized_max = (
                    1.0
                    if np.max(images[i]) == max_abs
                    else (np.max(images[i]) - (-max_abs)) / (2 * max_abs)
                )
                images[i] = scale(images[i], [normalized_min + eps, normalized_max - eps])
                scale_suff = "_local_sym"
            else:
                images[i] = scale(images[i], [eps, 1.0 - eps])
                scale_suff = "_local_unsym"

    surrounding_value = kwargs["surrounding_value"] if "surrounding_value" in kwargs else 0.0
    pad_value = kwargs["pad_value"] if "pad_value" in kwargs else 1.0
    kwargs["images"] = images
    kwargs["surrounding_value"] = surrounding_value  # type: ignore
    kwargs["pad_value"] = pad_value  # type: ignore
    grid = make_grid(**kwargs)  # type: ignore
    assert isinstance(surrounding_value, float)  # to make mypy happy
    assert isinstance(pad_value, float)  # to make mypy happy
    cmap = _stack_with_black_white(surrounding_value, pad_value, eps, 1.0 - eps, cmap)
    vmin = surrounding_value  # for readability
    vmax = pad_value  # for readability

    return grid, cmap, vmin, vmax, scale_suff


def save_grid(png_file, **kwargs):
    # type: (np.ndarray, str, Dict) -> None
    """Save image grid as PNG
    :param png_file: Write image to this PNG file
    :param kwargs: For further arguments, see docs of
                   `make_grid_with_black_boxes_and_white_background`

    Converted from PyTorch to Numpy and slightly modified based on
    https://pytorch.org/vision/stable/_modules/torchvision/utils.html#save_image

    For LICENSING and COPYRIGHT for this function see pytorch/vision's license at:
    https://github.com/pytorch/vision/blob/main/LICENSE
    """
    assert "images" in kwargs
    grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
        **kwargs  # type: ignore
    )

    png_file = png_file.replace(".png", "{}.png".format(scale_suff))
    plt.imsave(png_file, np.squeeze(grid), cmap=cmap, vmin=vmin, vmax=vmax)
    print("Wrote " + png_file)
