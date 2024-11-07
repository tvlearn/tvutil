# -*- coding: utf-8 -*-
# Copyright (C) 2020 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import print_function

import numpy as np
from numpy import ndarray
from scipy.stats import multivariate_normal
from skimage.util.shape import view_as_windows
from typing import Union, Callable, Tuple, List  # noqa

try:
    import torch as to
    from torch import Tensor  # noqa
except ImportError:
    pass
import time

import cppUtils


def variance_merger(values):
    # type: (Union[Tensor, ndarray]) -> Union[Tensor, ndarray]
    """Merge data estimates by taking the variance.

    :param values: see `mean_merger` docs
    :return: variance of `values`, scalar
    """
    to_or_np = np if isinstance(values, ndarray) else to
    return to_or_np.var(values)


def weighted_mean_merger(values, height, width, inds_relevant):
    # type: (Union[Tensor, ndarray], int, int, Union[Tensor, ndarray]) -> Union[Tensor, ndarray]
    """Merge data estimates by taking a weighted mean.

    :param values: see `mean_merger` docs
    :param height: patch height
    :param width: patch width
    :param inds_relevant: Indices of relevant pixels (e.g. [0] for top-left corner patch)
    :return: weighted mean of `values`, scalar
    """
    weights = gaussian2d(height, width)[0].flatten()[inds_relevant]
    weights /= weights.sum()
    weights = (
        weights
        if isinstance(values, ndarray)
        else to.from_numpy(weights).to(dtype=values.dtype, device=values.device)
    )
    return (values * weights).sum()


def gaussian2d(
    no_bins_dim1,
    no_bins_dim2,
    lim_dim1=[-1.0, 1.0],
    lim_dim2=[-1.0, 1.0],
    mu=0.0,
    sigma=1.0,
):
    # type: (int, int, List[float], List[float], float, float) -> Tuple[ndarray, ndarray, ndarray]
    """Returns the pdf of a two-dimensional multivariate Gaussian distribution.

    :param no_bins_dim1: grid size in first dimension
    :param no_bins_dim2: grid size in second dimension
    :param lim_dim1: domain limits of pdf in first direction
    :param lim_dim2: domain limits of pdf in second direction
    :param mu: mean of pdf
    :param sigma: variance of pdf
    :param device: torch.device of output Tensor, defaults to to.device('cpu').
    :return: tuple containing (pdf values, first dimension grid, second dimension grid)
    """
    step_dim1 = np.diff(lim_dim1) / no_bins_dim1
    step_dim2 = np.diff(lim_dim2) / no_bins_dim2

    grd = np.empty((no_bins_dim1, no_bins_dim2, 2))
    grd[:, :, 0], grd[:, :, 1] = np.mgrid[
        lim_dim1[0] : lim_dim1[1] : step_dim1, lim_dim2[0] : lim_dim2[1] : step_dim2  # type: ignore
    ]  # type: ignore
    pdf = multivariate_normal.pdf(grd, mu * np.ones(2), sigma * np.eye(2))  # is (2, 2)
    return pdf, grd[:, :, 0], grd[:, :, 1]


def vprint(message, verbose=True, **kwargs):
    if verbose:
        print(message, **kwargs)


class OverlappingNDPatches:
    def __init__(
        self,
        image,
        patch_shapes,
        patch_shift,
        verbose=False,
    ):
        # type: (Union[Tensor, ndarray], int, int, int, bool) -> None
        """Back and forth transformation for image segmentation into overlapping patches.
        Makes use of `skimage.util.view_as_windows`.

        :param image: Tensor to be cut into patches and reconstructed.
        :param patch_shapes: Will be passed as `window_shape` to `skimage.util.view_as_windows`.
        :param patch_shift: Will be passed as `step` to `skimage.util.view_as_windows`.
        :param verbose: Whether to print details when merging patches
        """
        assert len(patch_shapes) == len(
            image.shape
        ), "length of patch shapes must match length of image shape"
        patch_shift_greater_than_patch_shapes = patch_shift > np.array(patch_shapes)
        if (patch_shift_greater_than_patch_shapes).any():
            positions = np.nonzero(patch_shift_greater_than_patch_shapes)[0]
            vprint(
                f"WARNING: The patch shift is greater than the patch shapes at position {positions}!",
                flush=True,
                verbose=verbose,
            )
            vprint("Some image pixels will not be reconstructed!", flush=True, verbose=verbose)

        self._torch = False if isinstance(image, ndarray) else True
        self._verbose = verbose
        self._patch_shapes = patch_shapes
        self._patch_shift = patch_shift
        self.device, self.precision = (
            None if isinstance(image, ndarray) else image.device
        ), image.dtype
        image_np = image if isinstance(image, ndarray) else image.detach().cpu().numpy()

        # Infer some parameters
        vprint("Infer Parameters...", end="", flush=True, verbose=verbose)
        start = time.monotonic()
        params = self.get_parameters(image_np, patch_shapes, patch_shift)
        vprint(f"Done in {time.monotonic() - start:.2f} s", flush=True, verbose=verbose)

        # cut patches
        vprint("Extracting patches...", end="", flush=True, verbose=verbose)
        start = time.monotonic()
        patches, patches_np_not_isnan = self.extract_patches(
            image, patch_shapes, patch_shift, **params
        )
        vprint(f"Done in {time.monotonic() - start:.2f} s", flush=True, verbose=verbose)

        vprint("Initialize back-transformation...", end="", flush=True, verbose=verbose)
        self._cpp = cppUtils.OverlappingPatches(
            params["no_pixels_to_synthesize"],
            patch_shift,
            params["ind_to_synthesize"],
            np.array(image.shape),
            np.array(patch_shapes),
            np.array(params["no_patches_per_axis"]),
            np.array(params["no_patches_per_axis_shift_1"]),
        )
        vprint(f"Done in {time.monotonic() - start:.2f} s", flush=True, verbose=verbose)

        self._image, self._patches = image, patches

    def get_parameters(self, image, patch_shapes, patch_shift):

        # infer some parameters
        image_not_incomplete = np.logical_not(np.isnan(image).any())

        no_pixels_in_patch = np.prod(patch_shapes)

        no_patches_per_axis = [
            int(np.ceil(float(image_shp - patch_shp) / patch_shift) + 1)
            for image_shp, patch_shp in zip(image.shape, patch_shapes)
        ]
        no_patches = np.prod(no_patches_per_axis)

        no_patches_per_axis_shift_1 = [
            int(np.ceil(float(image_shp - patch_shp)) + 1)
            for image_shp, patch_shp in zip(image.shape, patch_shapes)
        ]
        no_patches_shift_1 = np.prod(no_patches_per_axis_shift_1)  # no patches for step=1

        to_be_synthesized = (
            np.isnan(image) if np.isnan(image).any() else np.ones_like(image, dtype=bool)
        )  # indicates which pixels of the input image are to be reconstructed
        ind_to_synthesize = np.array(np.nonzero(to_be_synthesized))
        no_pixels_to_synthesize = ind_to_synthesize[0].size  # no missing values
        return dict(
            image_not_incomplete=image_not_incomplete,
            no_pixels_in_patch=no_pixels_in_patch,
            no_patches_per_axis=no_patches_per_axis,
            no_patches=no_patches,
            no_patches_per_axis_shift_1=no_patches_per_axis_shift_1,
            no_patches_shift_1=no_patches_shift_1,
            to_be_synthesized=to_be_synthesized,
            ind_to_synthesize=ind_to_synthesize,
            no_pixels_to_synthesize=no_pixels_to_synthesize,
        )

    def extract_patches(
        self,
        image,
        patch_shapes,
        patch_shift,
        no_patches,
        no_pixels_in_patch,
        no_patches_per_axis_shift_1,
        **kwargs,
    ):
        patches_np = view_as_windows(
            image, window_shape=patch_shapes, step=1
        )  # moves sliding window left->right and then top->bottom
        # is (image_height-patch_height+1, image_width-patch_width+1, patch_height, patch_width)
        if patch_shift > 1:
            inds = [
                (
                    np.append(
                        np.arange(1, no_patch, patch_shift),
                        [no_patch],
                    )
                    - 1
                ).flatten()
                for no_patch in no_patches_per_axis_shift_1
            ]  # indices of relevant patches for step=patch_shift in vertical direction
            patches_np = patches_np[np.ix_(*inds)]  # TODO: Find better way to do this?

        patches_np = patches_np.reshape(no_patches, no_pixels_in_patch)
        patches_np_not_isnan = np.logical_not(np.isnan(patches_np))
        return np.ascontiguousarray(patches_np), patches_np_not_isnan

    def get_image_shape(self):
        # type: () -> Tuple[int, int]
        """Return shape of input image

        :return: Image shape
        """
        return tuple(self._image.shape)  # type: ignore

    def get_number_of_patches(self, discard_empty=True):
        # type: (bool) -> int
        """Return number of patches cut from image

        :param discard_empty: Whether to discard patches that do not contain finite entries
        :return: Number of patches
        """
        to_or_np = to if self._torch else np
        not_isnan = to_or_np.logical_not(to_or_np.isnan(self._patches))  # type: ignore
        no_patches_with_discarding = to_or_np.sum(
            not_isnan.any(**{"dim" if self._torch else "axis": 1})
        ).item()
        no_patches_without_discarding = int(self._patches.shape[0])
        return no_patches_with_discarding if discard_empty else no_patches_without_discarding

    def get_patch_shape_shift(self):
        # type: () -> Tuple[int, int, int]
        """Return the patch height, width and shift

        :return: Tuple with (patch height, width and shift)
        """
        return *self._patch_shapes, self._patch_shift

    def get(self, discard_empty=True):
        # type: (bool) -> Union[Tensor, ndarray]
        """Returns patches cut from image.

        :param discard_empty: Whether to discard patches that do not contain finite entries
        :return: Image patches tensor, is (no_pixels_per_patch, no_patches)
        """
        to_or_np = to if self._torch else np
        patches = to.from_numpy(self._patches) if self._torch else self._patches
        if to_or_np.logical_not(to_or_np.isnan(self._patches).any()):
            return patches
        else:
            if discard_empty:
                not_isnan = to_or_np.logical_not(to_or_np.isnan(self._patches))  # type: ignore
                inds_not_empty = not_isnan.any(**{"dim" if self._torch else "axis": 0})
                return patches[:, inds_not_empty]
            else:
                return patches

    def set(self, new_patches, discarded_empty=True):
        # type: (Union[Tensor, ndarray], bool) -> None
        """Update image patches tensor to new values

        :param new_patches: Image patches tensor filled with new values. `self._patches` will
                            be updated to this tensor, must be (no_pixels_per_patch, no_patches).
        :param discarded_empty: Whether patches without finite entries have been discarded when
                                `get` was called (compare docs of `get`).
        """
        to_or_np = to if self._torch else np
        new_patches = new_patches.numpy() if self._torch else new_patches
        if to_or_np.logical_not(to_or_np.isnan(self._patches).any()):
            assert (
                new_patches.shape == self._patches.shape
            ), "shape of new and internal patches does not match"
            self._patches[:, :] = new_patches
        else:
            if discarded_empty:
                not_isnan = to_or_np.logical_not(to_or_np.isnan(self._patches))  # type: ignore
                inds_not_empty = not_isnan.any(**{"dim" if self._torch else "axis": 0})
                assert (
                    new_patches.shape == self._patches[:, inds_not_empty].shape
                ), "shape of new and non-empty internal patches does not match"
                self._patches[:, inds_not_empty] = new_patches
            else:
                assert (
                    new_patches.shape == self._patches.shape
                ), "shape of new and internal patches does not match"
                self._patches[:, :] = new_patches

    def merge(self, merge_method: str = "mean"):
        # type: (Callable) -> Union[Tensor, ndarray]
        """Merge patches to obtain new image.

        :param merge_method: Function defining how pixel estimates from different patches are to be
                             merged, defaults to unweighted averaging.
        :return: Image obtained through patch averaging, is (height, width)
        """
        vprint("Merge patches...", end="", flush=True, verbose=self._verbose)
        assert merge_method in ("mean", "median", "max", "min", "variance")
        new_image = self._image.copy() if isinstance(self._image, ndarray) else self._image.clone()

        start = time.monotonic()
        self._cpp.merge(self._patches, new_image.reshape(-1), merge_method)
        vprint(f"Done in {time.monotonic() - start:.2f} s", flush=True, verbose=self._verbose)

        return new_image

    def set_and_merge(
        self,
        new_patches,
        discarded_empty=True,
        merge_method: str = "mean",
    ):
        # type: (Union[Tensor, ndarray], bool, Callable) -> Union[Tensor, ndarray]
        """Sequentially calls `set` and `merge`.

        :param new_patches: see docs of `set`
        :param discarded_empty: see docs of `set`
        :param merge_method: see docs of `merge`
        :return: see docs of `merge`
        """
        self.set(new_patches, discarded_empty)
        return self.merge(merge_method)
