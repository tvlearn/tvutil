# -*- coding: utf-8 -*-
# Copyright (C) 2020 Machine Learning Group of the University of Oldenburg.

import pytest
import numpy as np

try:
    import os
    import torch as to

    _device = (
        to.device("cpu")
        if "TVUTIL_GPU" not in os.environ
        else to.device("cuda:{}".format(int(os.environ["TVUTIL_GPU"])))
    )
except ImportError:
    _device = None

from tvutil.prepost import OverlappingPatches, MultiDimOverlappingPatches


@pytest.fixture(scope="function", params=[pytest.param(_device, marks=pytest.mark.gpu)])
def setup(request):
    class Setup:
        patch_height, patch_width, patch_shift = 2, 2, 1
        image = np.array(
            [[0.23362892, 0.4054638, 0.2431283], [0.8939867, 0.63614618, 0.69853074]],
            dtype=np.float64,
        )
        image_incomplete = image.copy()
        image_incomplete[1, 2] = float("nan")
        using_np = request.param is None
        if not using_np:
            assert isinstance(request.param, to.device)
            to_kwargs = {"dtype": to.float64, "device": request.param}
            image = to.from_numpy(image).to(**to_kwargs)
            image_incomplete = to.from_numpy(image_incomplete).to(**to_kwargs)
        tensor_type = np.ndarray if using_np else to.Tensor
        logical_not_fn = np.logical_not if using_np else to.logical_not
        isnan_fn = np.isnan if using_np else to.isnan
        arange_fn = np.arange if using_np else to.arange
        prints_enabled = True

        @staticmethod
        def repeat_fn(x, reps):
            return np.tile(x[:, :, None], reps) if Setup.using_np else x[:, :, None].repeat(*reps)

    return Setup


def test_get_image_shape(setup):
    OLP = OverlappingPatches(setup.image, setup.patch_height, setup.patch_width, setup.patch_shift)
    ndims = 4
    MultiDimOLP = MultiDimOverlappingPatches(
        setup.repeat_fn(setup.image, (1, 1, ndims)),
        setup.patch_height,
        setup.patch_width,
        setup.patch_shift,
    )
    image_shape = setup.image.shape
    assert OLP.get_image_shape() == tuple(image_shape)
    assert MultiDimOLP.get_image_shape() == tuple(image_shape) + (ndims,)


def test_get_number_of_patches(setup):
    patch_shift = 1
    ndims = 4
    OLP = OverlappingPatches(setup.image, setup.patch_height, setup.patch_width, patch_shift)
    MultiDimOLP = MultiDimOverlappingPatches(
        setup.repeat_fn(setup.image, (1, 1, ndims)),
        setup.patch_height,
        setup.patch_width,
        patch_shift,
    )
    no_patches_1dim = (setup.image.shape[0] - setup.patch_height + 1) * (
        setup.image.shape[1] - setup.patch_width + 1
    )
    assert OLP.get_number_of_patches() == no_patches_1dim
    assert MultiDimOLP.get_number_of_patches() == no_patches_1dim


def test_get_patch_height_width_shift(setup):
    patch_shift = 1
    ndims = 4
    OLP = OverlappingPatches(setup.image, setup.patch_height, setup.patch_width, patch_shift)
    MultiDimOLP = MultiDimOverlappingPatches(
        setup.repeat_fn(setup.image, (1, 1, ndims)),
        setup.patch_height,
        setup.patch_width,
        patch_shift,
    )
    assert OLP.get_patch_height_width_shift() == (
        setup.patch_height,
        setup.patch_width,
        patch_shift,
    )
    assert MultiDimOLP.get_patch_height_width_shift() == (
        setup.patch_height,
        setup.patch_width,
        patch_shift,
    )


def test_get(setup):
    OLP = OverlappingPatches(setup.image, setup.patch_height, setup.patch_width, setup.patch_shift)
    OLP_incomplete = OverlappingPatches(
        setup.image_incomplete, setup.patch_height, setup.patch_width, setup.patch_shift
    )
    image_shape, patch_height, patch_width = (
        setup.image.shape,
        setup.patch_height,
        setup.patch_width,
    )
    patches, patches_incomplete = OLP.get(), OLP_incomplete.get()

    assert isinstance(patches, setup.tensor_type)
    assert setup.logical_not_fn(setup.isnan_fn(patches).any())
    assert isinstance(patches_incomplete, setup.tensor_type)
    assert setup.isnan_fn(patches_incomplete).any()

    no_patches = (image_shape[0] - patch_height + 1) * (image_shape[1] - patch_width + 1)
    no_pixels = patch_height * patch_width
    assert patches.shape == (no_pixels, no_patches)
    assert patches_incomplete.shape == (no_pixels, no_patches)

    if setup.prints_enabled:
        print(setup.image)
        print(patches)


def test_set(setup):
    OLP = OverlappingPatches(setup.image, setup.patch_height, setup.patch_width, setup.patch_shift)
    OLP_incomplete = OverlappingPatches(
        setup.image_incomplete,
        setup.patch_height,
        setup.patch_width,
        setup.patch_shift,
    )
    copy_fn = np.copy if setup.using_np else to.clone
    patches, patches_incomplete = copy_fn(OLP.get()), copy_fn(OLP_incomplete.get())

    no_pixels = patches.size if setup.using_np else patches.numel()
    new_patches = setup.arange_fn(no_pixels, dtype=patches.dtype).reshape(patches.shape)
    new_patches_incomplete = copy_fn(patches_incomplete)
    new_patches_incomplete[setup.isnan_fn(patches_incomplete)] = 0.0
    OLP.set(new_patches)
    OLP_incomplete.set(new_patches_incomplete)
    allclose_fn = np.allclose if setup.using_np else to.allclose
    assert not allclose_fn(patches, OLP.get())
    assert allclose_fn(new_patches, OLP.get())
    assert not allclose_fn(
        patches[setup.isnan_fn(patches_incomplete)],
        OLP_incomplete.get()[setup.isnan_fn(patches_incomplete)],
    )
    assert allclose_fn(new_patches_incomplete, OLP_incomplete.get())

    if setup.prints_enabled:
        print(OLP.get())
        print(OLP_incomplete.get())


def test_merge(setup):
    image, image_incomplete = setup.image, setup.image_incomplete
    OLP = OverlappingPatches(setup.image, setup.patch_height, setup.patch_width, setup.patch_shift)
    OLP_incomplete = OverlappingPatches(
        setup.image_incomplete,
        setup.patch_height,
        setup.patch_width,
        setup.patch_shift,
    )
    copy_fn = np.copy if setup.using_np else to.clone
    patches, patches_incomplete = copy_fn(OLP.get()), copy_fn(OLP_incomplete.get())

    _image = OLP.merge()
    allclose_fn = np.allclose if setup.using_np else to.allclose
    assert allclose_fn(image, _image)

    no_pixels = patches.size if setup.using_np else patches.numel()
    new_patches = setup.arange_fn(no_pixels, dtype=patches.dtype).reshape(patches.shape)
    new_patches_incomplete = copy_fn(patches_incomplete)
    new_patches_incomplete[setup.isnan_fn(patches_incomplete)] = 0.0
    new_image = OLP.set_and_merge(new_patches)
    new_image_incomplete = OLP_incomplete.set_and_merge(new_patches_incomplete)

    assert not allclose_fn(image, new_image)
    assert allclose_fn(
        image[setup.logical_not_fn(setup.isnan_fn(image_incomplete))],
        new_image_incomplete[setup.logical_not_fn(setup.isnan_fn(image_incomplete))],
    )
    assert not allclose_fn(
        image[setup.isnan_fn(image_incomplete)],
        new_image_incomplete[setup.isnan_fn(image_incomplete)],
    )

    if setup.prints_enabled:
        print(image)
        print(new_patches)
        print(new_image)
        print()
        print(image)
        print(new_patches_incomplete)
        print(new_image_incomplete)


def test_get_ndim_with_concatenation(setup):
    OLP = OverlappingPatches(setup.image, setup.patch_height, setup.patch_width, setup.patch_shift)
    patches = OLP.get()

    ndims = 4
    multidim_image = setup.repeat_fn(setup.image, (1, 1, ndims))
    multidim_image_incomplete = setup.repeat_fn(setup.image_incomplete, (1, 1, ndims))
    MultiDimOLP = MultiDimOverlappingPatches(
        multidim_image, setup.patch_height, setup.patch_width, setup.patch_shift
    )
    MultiDimOLP_incomplete = MultiDimOverlappingPatches(
        multidim_image_incomplete,
        setup.patch_height,
        setup.patch_width,
        setup.patch_shift,
    )
    multidim_patches = MultiDimOLP.get()
    multidim_patches_incomplete = MultiDimOLP_incomplete.get()
    assert len(multidim_patches.shape) == 2
    assert (multidim_patches.shape[0] // ndims) == patches.shape[0]
    assert multidim_patches.shape[1] == patches.shape[1]
    if not setup.using_np:
        assert multidim_patches.device == patches.device
    assert multidim_patches.dtype == patches.dtype
    assert not setup.isnan_fn(multidim_patches).any()
    assert multidim_patches.shape == multidim_patches_incomplete.shape
    assert setup.isnan_fn(multidim_patches_incomplete).any()


def test_get_ndim_no_concatenation(setup):
    OLP = OverlappingPatches(setup.image, setup.patch_height, setup.patch_width, setup.patch_shift)
    patches = OLP.get()

    ndims = 4

    multidim_image = setup.repeat_fn(setup.image, (1, 1, ndims))
    multidim_image_incomplete = setup.repeat_fn(setup.image_incomplete, (1, 1, ndims))
    MultiDimOLP = MultiDimOverlappingPatches(
        multidim_image, setup.patch_height, setup.patch_width, setup.patch_shift
    )
    MultiDimOLP_incomplete = MultiDimOverlappingPatches(
        multidim_image_incomplete,
        setup.patch_height,
        setup.patch_width,
        setup.patch_shift,
    )
    multidim_patches = MultiDimOLP.get(concatenate=False)
    multidim_patches_incomplete = MultiDimOLP_incomplete.get(concatenate=False)
    assert multidim_patches.shape[:2] == patches.shape
    assert multidim_patches.shape[2] == ndims
    if not setup.using_np:
        assert multidim_patches.device == patches.device
    assert multidim_patches.dtype == patches.dtype
    assert not setup.isnan_fn(multidim_patches).any()
    assert multidim_patches.shape == multidim_patches_incomplete.shape
    assert setup.isnan_fn(multidim_patches_incomplete).any()
    allclose_fn = np.allclose if setup.using_np else to.allclose
    for ind_dim in range(ndims):
        assert allclose_fn(multidim_patches[:, :, ind_dim], patches)


def test_set_ndim_no_concatenation(setup):
    ndims = 4

    multidim_image = setup.repeat_fn(setup.image, (1, 1, ndims))
    multidim_image_incomplete = setup.repeat_fn(setup.image_incomplete, (1, 1, ndims))
    MultiDimOLP = MultiDimOverlappingPatches(
        multidim_image, setup.patch_height, setup.patch_width, setup.patch_shift
    )
    MultiDimOLP_incomplete = MultiDimOverlappingPatches(
        multidim_image_incomplete, setup.patch_height, setup.patch_width, setup.patch_shift
    )
    multidim_patches = MultiDimOLP.get(concatenate=False)
    multidim_patches_incomplete = MultiDimOLP_incomplete.get(concatenate=False)

    no_pixels = multidim_patches.size if setup.using_np else multidim_patches.numel()
    new_multidim_patches = setup.arange_fn(no_pixels, dtype=multidim_patches.dtype).reshape(
        multidim_patches.shape
    )
    copy_fn = np.copy if setup.using_np else to.clone
    new_multidim_patches_incomplete = copy_fn(multidim_patches_incomplete)
    new_multidim_patches_incomplete[setup.isnan_fn(multidim_patches_incomplete)] = 0.0
    MultiDimOLP.set(new_multidim_patches, concatenated=False)
    MultiDimOLP_incomplete.set(new_multidim_patches_incomplete, concatenated=False)
    assert multidim_patches.shape == MultiDimOLP.get(concatenate=False).shape
    allclose_fn = np.allclose if setup.using_np else to.allclose
    assert not allclose_fn(multidim_patches, MultiDimOLP.get(concatenate=False))
    assert allclose_fn(new_multidim_patches, MultiDimOLP.get(concatenate=False))
    assert not allclose_fn(
        multidim_patches[setup.isnan_fn(multidim_patches_incomplete)],
        MultiDimOLP_incomplete.get(concatenate=False)[setup.isnan_fn(multidim_patches_incomplete)],
    )
    assert allclose_fn(
        new_multidim_patches_incomplete, MultiDimOLP_incomplete.get(concatenate=False)
    )

    if setup.prints_enabled:
        print(MultiDimOLP.get(concatenate=False))
        print(MultiDimOLP_incomplete.get(concatenate=False))


def test_merge_ndim_no_concatenation(setup):
    ndims = 4

    multidim_image = setup.repeat_fn(setup.image, (1, 1, ndims))
    multidim_image_incomplete = setup.repeat_fn(setup.image_incomplete, (1, 1, ndims))
    MultiDimOLP = MultiDimOverlappingPatches(
        multidim_image, setup.patch_height, setup.patch_width, setup.patch_shift
    )
    MultiDimOLP_incomplete = MultiDimOverlappingPatches(
        multidim_image_incomplete,
        setup.patch_height,
        setup.patch_width,
        setup.patch_shift,
    )
    copy_fn = np.copy if setup.using_np else to.clone
    multidim_patches, multidim_patches_incomplete = (
        copy_fn(MultiDimOLP.get(concatenate=False)),
        copy_fn(MultiDimOLP_incomplete.get(concatenate=False)),
    )

    _multidim_image = MultiDimOLP.merge()
    allclose_fn = np.allclose if setup.using_np else to.allclose
    assert allclose_fn(multidim_image, _multidim_image)

    no_pixels = multidim_patches.size if setup.using_np else multidim_patches.numel()
    new_multidim_patches = setup.arange_fn(no_pixels, dtype=multidim_patches.dtype).reshape(
        multidim_patches.shape
    )
    new_multidim_patches_incomplete = copy_fn(multidim_patches_incomplete)
    new_multidim_patches_incomplete[setup.isnan_fn(multidim_patches_incomplete)] = 0.0
    new_multidim_image = MultiDimOLP.set_and_merge(new_multidim_patches, concatenated=False)
    new_multidim_image_incomplete = MultiDimOLP_incomplete.set_and_merge(
        new_multidim_patches_incomplete, concatenated=False
    )

    assert new_multidim_image.shape == multidim_image.shape
    assert new_multidim_image_incomplete.shape == multidim_image.shape
    assert not allclose_fn(multidim_image, new_multidim_image)
    assert allclose_fn(
        multidim_image[setup.logical_not_fn(setup.isnan_fn(multidim_image_incomplete))],
        new_multidim_image_incomplete[
            setup.logical_not_fn(setup.isnan_fn(multidim_image_incomplete))
        ],
    )
    assert not allclose_fn(
        multidim_image[setup.isnan_fn(multidim_image_incomplete)],
        new_multidim_image_incomplete[setup.isnan_fn(multidim_image_incomplete)],
    )

    if setup.prints_enabled:
        print(multidim_image)
        print(new_multidim_patches)
        print(new_multidim_image)
        print()
        print(multidim_image)
        print(new_multidim_patches_incomplete)
        print(new_multidim_image_incomplete)
