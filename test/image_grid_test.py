# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.

import pytest
import numpy as np

from tvutil.viz import make_grid, scale, make_grid_with_black_boxes_and_white_background


@pytest.fixture(scope="module")
def setup():
    class Setup:
        no_images = 10
        no_channels = 3
        height = 64
        width = height
        nrow = 4
        ncol = int(np.ceil(float(no_images) / nrow))
        images = np.random.random((no_images, no_channels, height, width))

    return Setup


def test_make_grid_without_surrounding_without_padding_without_repeat(setup):
    grid = make_grid(setup.images, nrow=setup.nrow, surrounding=0, padding=0)

    summed_and_padded_heights = setup.ncol * setup.height
    summed_and_padded_widths = setup.nrow * setup.width
    assert grid.shape == (setup.no_channels, summed_and_padded_heights, summed_and_padded_widths)


def test_make_grid_without_surrounding_without_padding_with_repeat(setup):
    repeat = 3
    grid = make_grid(setup.images, nrow=setup.nrow, surrounding=0, padding=0, repeat=repeat)
    summed_and_padded_heights = setup.ncol * setup.height * repeat
    summed_and_padded_widths = setup.nrow * setup.width * repeat
    assert grid.shape == (setup.no_channels, summed_and_padded_heights, summed_and_padded_widths)


def test_make_grid_with_surrounding_without_padding_without_repeat(setup):
    surrounding = 2
    grid = make_grid(setup.images, nrow=setup.nrow, surrounding=surrounding, padding=0)
    summed_and_padded_heights = setup.ncol * (setup.height + 2 * surrounding)
    summed_and_padded_widths = setup.nrow * (setup.width + 2 * surrounding)
    assert grid.shape == (setup.no_channels, summed_and_padded_heights, summed_and_padded_widths)


def test_make_grid_without_surrounding_with_padding_without_repeat(setup):
    padding = 1
    grid = make_grid(setup.images, nrow=setup.nrow, surrounding=0, padding=padding)
    summed_and_padded_heights = setup.ncol * (setup.height + padding) + padding
    summed_and_padded_widths = setup.nrow * (setup.width + padding) + padding
    assert grid.shape == (setup.no_channels, summed_and_padded_heights, summed_and_padded_widths)


def test_make_grid_with_surrounding_with_padding_with_repeat(setup):
    padding, surrounding, repeat = 2, 1, 3
    grid = make_grid(
        setup.images, nrow=setup.nrow, surrounding=surrounding, padding=padding, repeat=repeat
    )
    summed_and_padded_heights = (
        setup.ncol * (setup.height * repeat + 2 * surrounding + padding) + padding
    )
    summed_and_padded_widths = (
        setup.nrow * (setup.width * repeat + 2 * surrounding + padding) + padding
    )
    assert grid.shape == (setup.no_channels, summed_and_padded_heights, summed_and_padded_widths)


def test_make_grid_with_black_boxes_and_white_background(setup):
    padding, surrounding, repeat = 2, 1, 3
    eps = 0.02
    grid, cmap, vmin, vmax, _ = make_grid_with_black_boxes_and_white_background(
        images=setup.images,
        nrow=setup.nrow,
        surrounding=surrounding,
        padding=padding,
        repeat=repeat,
        eps=eps,
    )
    summed_and_padded_heights = (
        setup.ncol * (setup.height * repeat + 2 * surrounding + padding) + padding
    )
    summed_and_padded_widths = (
        setup.nrow * (setup.width * repeat + 2 * surrounding + padding) + padding
    )
    assert grid.shape == (setup.no_channels, summed_and_padded_heights, summed_and_padded_widths)
    assert np.min(grid) == 0.0
    assert np.max(grid) == 1.0
    assert cmap.name == "jet-bw"


def test_scale(setup):
    original = setup.images
    new_range = [12.0, 14.5]
    scaled = scale(original, new_range=new_range)
    assert np.min(scaled) == new_range[0]
    assert np.max(scaled) == new_range[1]
