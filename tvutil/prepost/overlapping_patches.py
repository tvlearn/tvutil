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


def mean_merger(values):
    # type: (Union[Tensor, ndarray]) -> Union[Tensor, ndarray]
    """Merge data estimates using mean.

    :param values: estimates from overlapping patches for given pixel in reconstructed image
    :return: mean of `values`, scalar
    """
    to_or_np = np if isinstance(values, ndarray) else to
    return to_or_np.mean(values)


def max_merger(values):
    # type: (Union[Tensor, ndarray]) -> Union[Tensor, ndarray]
    """Merge data estimates by taking the maximum value.

    :param values: see `mean_merger` docs
    :return: max of `values`, scalar
    """
    to_or_np = np if isinstance(values, ndarray) else to
    return to_or_np.max(values)


def min_merger(values):
    # type: (Union[Tensor, ndarray]) -> Union[Tensor, ndarray]
    """Merge data estimates by taking the minimum value.

    :param values: see `mean_merger` docs
    :return: min of `values`, scalar
    """
    to_or_np = np if isinstance(values, ndarray) else to
    return to_or_np.min(values)


def median_merger(values):
    # type: (Union[Tensor, ndarray]) -> Union[Tensor, ndarray]
    """Merge data estimates by taking the median.

    :param values: see `mean_merger` docs
    :return: median of `values`, scalar
    """
    to_or_np = np if isinstance(values, ndarray) else to
    return to_or_np.median(values)


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


class OverlappingPatches:
    def __init__(
        self,
        image,
        patch_height,
        patch_width,
        patch_shift,
        verbose=False,
    ):
        # type: (Union[Tensor, ndarray], int, int, int, bool) -> None
        """Back and forth transformation for image segmentation into overlapping patches.
        Makes use of `skimage.util.view_as_windows`.

        :param image: Tensor to be cut into patches and reconstructed. Must be 2-dimensional,
                      (height, width).
        :param patch_height: Will be passed as `window_shape[0]` to `skimage.util.view_as_windows`.
        :param patch_width: Will be passed as `window_shape[1]` to `skimage.util.view_as_windows`.
        :param patch_shift: Will be passed as `step` to `skimage.util.view_as_windows`.
        :param verbose: Whether to print details when merging patches
        """
        assert np.ndim(image) == 2, "image tensor must be two-dimensional (width x height)"

        self._torch = False if isinstance(image, ndarray) else True
        self._verbose = verbose
        self._patch_height = patch_height
        self._patch_width = patch_width
        self._patch_shift = patch_shift
        device, precision = None if isinstance(image, ndarray) else image.device, image.dtype
        image_np = image if isinstance(image, ndarray) else image.detach().cpu().numpy()

        # infer some parameters
        image_not_incomplete = np.logical_not(np.isnan(image_np).any())
        image_height, image_width = image_np.shape[0], image_np.shape[1]
        no_pixels_in_patch = patch_height * patch_width
        no_patches_vert = int(
            np.ceil(float(image_height - patch_height) / patch_shift) + 1
        )  # no patches in vertical direction
        no_patches_horz = int(
            np.ceil(float(image_width - patch_width) / patch_shift) + 1
        )  # no patches in horizontal direction
        no_patches = no_patches_vert * no_patches_horz
        no_patches_vert_shift_1 = int(
            np.ceil(float(image_height - patch_height)) + 1
        )  # no patches in vertical dir for step=1
        no_patches_horz_shift_1 = int(
            np.ceil(float(image_width - patch_width)) + 1
        )  # no patches in horizontal dir for step=1
        no_patches_shift_1 = (
            no_patches_vert_shift_1 * no_patches_horz_shift_1
        )  # no patches for step=1

        ninds_ = np.arange(no_patches_shift_1).reshape(
            no_patches_vert_shift_1, no_patches_horz_shift_1
        )  # lower right patch locations in image for step = 1
        hinds = (
            np.unique(
                np.append(
                    np.arange(1, no_patches_vert_shift_1, patch_shift),
                    [no_patches_vert_shift_1],
                )
            )
            - 1
        ).flatten()  # indices of relevant patches for step=patch_shift in vertical direction
        winds = (
            np.unique(
                np.append(
                    np.arange(1, no_patches_horz_shift_1, patch_shift),
                    [no_patches_horz_shift_1],
                )
            )
            - 1
        ).flatten()  # indices of relevant patches for step=patch_shift in horizontal direction
        ninds = (ninds_[hinds, :][:, winds]).flatten()  # indices of relevant patches
        assert len(ninds) == no_patches
        dinds = np.arange(no_pixels_in_patch).reshape(
            patch_height, patch_width
        )  # spatial order of pixel indices, is (patch_height, patch_width),
        # indexed l->r and then top->bottom
        to_be_synthesized = (
            np.isnan(image_np) if np.isnan(image_np).any() else np.ones_like(image_np, dtype=bool)
        )  # indicates which pixels of the input image are to be reconstructed
        ind_rows_to_synthesize, ind_cols_to_synthesize = np.where(
            to_be_synthesized
        )  # index tuples of missing values, is (total # missing vals)
        no_pixels_to_synthesize = ind_rows_to_synthesize.size  # no missing values

        # cut patches
        print("Extracting patches...", end="")
        patches_np = view_as_windows(
            image_np, window_shape=[patch_height, patch_width], step=1
        )  # moves sliding window left->right and then top->bottom
        # is (image_height-patch_height+1, image_width-patch_width+1, patch_height, patch_width)
        patches_np = (
            patches_np.reshape(patch_height, patch_width, no_patches_shift_1)
            .reshape(no_patches_shift_1, no_pixels_in_patch)
            .T
        )  # is (no_pixels_in_patch,no_patches_shift_1)
        patches_np = patches_np[
            :, ninds
        ]  # remove patches not satisfying `patch_shift` is (no_pixels_in_patch,no_patches)
        patches_np_not_isnan = np.logical_not(np.isnan(patches_np))
        patches = (
            to.from_numpy(patches_np).to(dtype=precision, device=device)
            if self._torch
            else patches_np
        )
        print("Done")

        # compute indices required to merge patches back to image
        print("Initialize back-transformation...", end="")
        all_inds_relevant_patches = [0] * no_pixels_to_synthesize
        all_inds_relevant_values_in_patch = [0] * no_pixels_to_synthesize
        restorable = (
            np.zeros(no_pixels_to_synthesize, dtype=bool) if np.isnan(patches_np).any() else None
        )
        for p in range(no_pixels_to_synthesize):

            # location of missing value in original image
            r, c = ind_rows_to_synthesize[p], ind_cols_to_synthesize[p]

            # location of relevant patches for patch_shift = 1(rows and columns of ninds_)
            r_ = (
                np.arange(
                    max(r - patch_height + 2, 1),
                    min(r + 1, no_patches_vert_shift_1) + 1,
                )
                - 1
            )
            c_ = np.arange(max(c - patch_width + 2, 1), min(c + 1, no_patches_horz_shift_1) + 1) - 1

            ns_ = ninds_[r_, :][
                :, c_
            ].flatten()  # is no relevant patches for given pixel in original
            ds_ = np.sort(dinds[r - r_, :][:, c - c_].flatten())[::-1]

            # only use patches compatible with given patch_shift
            if patch_shift > 1:
                nsinds = np.isin(ns_, ninds)
                inds_relevant_patches, inds_relevant_values_in_patch = (
                    ns_[nsinds],
                    ds_[nsinds],
                )
            else:
                inds_relevant_patches, inds_relevant_values_in_patch = ns_, ds_

            # indices considering remaining patches
            if patch_shift > 1:
                inds_relevant_patches = np.where(np.isin(ninds, inds_relevant_patches))[0]

            if image_not_incomplete:
                all_inds_relevant_patches[p] = inds_relevant_patches
                all_inds_relevant_values_in_patch[p] = inds_relevant_values_in_patch
            else:
                relevant_patches = patches_np_not_isnan[:, inds_relevant_patches]
                ind_nonempty_patches = relevant_patches.any(axis=0)
                if ind_nonempty_patches.any():
                    assert restorable is not None  # to make mypy happy
                    restorable[p] = True
                    # ind_nonempty_patches = relevant_patches.any(axis=0)
                    all_inds_relevant_patches[p] = inds_relevant_patches[ind_nonempty_patches]
                    all_inds_relevant_values_in_patch[p] = inds_relevant_values_in_patch[
                        ind_nonempty_patches
                    ]
        print("Done")

        self._image, self._patches = image, patches
        self._ind_rows_to_synthesize = (
            to.from_numpy(ind_rows_to_synthesize).to(dtype=to.int64, device=device)
            if self._torch
            else ind_rows_to_synthesize
        )
        self._ind_cols_to_synthesize = (
            to.from_numpy(ind_cols_to_synthesize).to(dtype=to.int64, device=device)
            if self._torch
            else ind_cols_to_synthesize
        )
        self._no_pixels_to_synthesize = no_pixels_to_synthesize
        self._restorable = (
            (
                to.from_numpy(restorable).to(dtype=to.bool, device=device)
                if self._torch
                else restorable
            )
            if restorable is not None
            else None
        )

        all_inds_relevant_patches = [
            x.copy() if isinstance(x, np.ndarray) else np.ndarray(x)  # type: ignore
            for x in all_inds_relevant_patches
        ]
        self._all_inds_relevant_patches = [
            to.from_numpy(x).to(dtype=to.int64, device=device) if self._torch else x
            for x in all_inds_relevant_patches
        ]

        all_inds_relevant_values_in_patch = [
            x.copy() if isinstance(x, np.ndarray) else np.ndarray(x)  # type: ignore
            for x in all_inds_relevant_values_in_patch
        ]
        self._all_inds_relevant_values_in_patch = [
            to.from_numpy(x).to(dtype=to.int64, device=device) if self._torch else x
            for x in all_inds_relevant_values_in_patch
        ]

    def get_image_shape(self):
        # type: () -> Tuple[int, int]
        """Return shape of input image

        :return: Image shape, (height, width)
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
            not_isnan.any(**{"dim" if self._torch else "axis": 0})
        ).item()
        no_patches_without_discarding = int(self._patches.shape[1])
        return no_patches_with_discarding if discard_empty else no_patches_without_discarding

    def get_patch_height_width_shift(self):
        # type: () -> Tuple[int, int, int]
        """Return the patch height, width and shift

        :return: Tuple with (patch height, width and shift)
        """
        return self._patch_height, self._patch_width, self._patch_shift

    def get(self, discard_empty=True):
        # type: (bool) -> Union[Tensor, ndarray]
        """Returns patches cut from image.

        :param discard_empty: Whether to discard patches that do not contain finite entries
        :return: Image patches tensor, is (no_pixels_per_patch, no_patches)
        """
        to_or_np = to if self._torch else np
        if to_or_np.logical_not(to_or_np.isnan(self._patches).any()):
            return self._patches
        else:
            if discard_empty:
                not_isnan = to_or_np.logical_not(to_or_np.isnan(self._patches))  # type: ignore
                inds_not_empty = not_isnan.any(**{"dim" if self._torch else "axis": 0})
                return self._patches[:, inds_not_empty]
            else:
                return self._patches

    def set(self, new_patches, discarded_empty=True):
        # type: (Union[Tensor, ndarray], bool) -> None
        """Update image patches tensor to new values

        :param new_patches: Image patches tensor filled with new values. `self._patches` will
                            be updated to this tensor, must be (no_pixels_per_patch, no_patches).
        :param discarded_empty: Whether patches without finite entries have been discarded when
                                `get` was called (compare docs of `get`).
        """
        to_or_np = to if self._torch else np
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

    def merge(self, merge_method=mean_merger):
        # type: (Callable) -> Union[Tensor, ndarray]
        """Merge patches to obtain new image.

        :param merge_method: Function defining how pixel estimates from different patches are to be
                             merged, defaults to unweighted averaging.
        :return: Image obtained through patch averaging, is (height, width)
        """
        new_image = self._image.copy() if isinstance(self._image, ndarray) else self._image.clone()
        for p in range(self._no_pixels_to_synthesize):
            if self._restorable is not None and not self._restorable[p]:
                continue

            r, c = self._ind_rows_to_synthesize[p], self._ind_cols_to_synthesize[p]
            inds_relevant_patches = self._all_inds_relevant_patches[p]
            inds_relevant_values_in_patch = self._all_inds_relevant_values_in_patch[p]

            restored = self._patches[inds_relevant_values_in_patch, inds_relevant_patches]

            if self._verbose:
                print("Processing image pixel at ({},{})".format(r, c))
                print("=" * 12)
                print("Estimates from all patches \n  {}\n".format(restored))

            kwargs = (
                {
                    "height": self._patch_height,
                    "width": self._patch_width,
                    "inds_relevant": inds_relevant_values_in_patch,
                }
                if merge_method == weighted_mean_merger
                else {}
            )
            estimate = merge_method(restored, **kwargs)

            if self._verbose:
                print("Merged estimate is \n  {}\n".format(estimate))

            new_image[r, c] = estimate

        return new_image

    def set_and_merge(
        self,
        new_patches,
        discarded_empty=True,
        merge_method=mean_merger,
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


class MultiDimOverlappingPatches:
    def __init__(
        self,
        image,
        patch_height,
        patch_width,
        patch_shift,
        verbose=False,
    ):
        # type: (Union[Tensor, ndarray], int, int, int, bool) -> None
        """Sequentially apply transformations implemented by `OverlappingPatches` for multi-channel
        data (e.g. RGB images).

        :param image: Tensor to be cut into patches and reconstructed, must be 3-dimensional,
                      (height, width, no_channels).
        :param patch_height: see `OverlappingPatches.__init__` docs
        :param patch_width: see `OverlappingPatches.__init__` docs
        :param patch_shift: see `OverlappingPatches.__init__` docs
        :param verbose: see `OverlappingPatches.__init__` docs
        """
        self._torch = False if isinstance(image, ndarray) else True
        no_dims = np.ndim(image) if isinstance(image, ndarray) else image.dim()
        assert no_dims == 3, "image tensor must be three-dimensional (width x height x no_channels)"
        self.no_channels = image.shape[2]
        self.OVPs = [
            OverlappingPatches(
                np.ascontiguousarray(image[:, :, ch]) if not self._torch else image[:, :, ch],
                patch_height,
                patch_width,
                patch_shift,
                verbose=verbose,
            )
            for ch in range(self.no_channels)
        ]
        breakpoint()

    def get_image_shape(self):
        # type: () -> Tuple[int, int, int]
        """Return image shape.

        :return: Image shape (height, width, no_channels)
        """
        return self.OVPs[0].get_image_shape() + (self.no_channels,)

    def get_number_of_patches(self, discard_empty=True):
        # type: (bool) -> int
        """Return number of image patches.

        :param discard_empty: see `OverlappingPatches.get_number_of_patches` docs
        :return: Number of patches, scalar
        """
        return self.OVPs[0].get_number_of_patches()

    def get_patch_height_width_shift(self):
        # type: () -> Tuple[int, int, int]
        """Return patch height, width and shift (cf. docs of
        `OverlappingPatches.get_patch_height_width_shift`).

        :return: Tuple with (patch_height, patch_width, patch_shift)
        """

        return self.OVPs[0].get_patch_height_width_shift()

    def get(self, discard_empty=False, concatenate=True):
        # type: (bool, bool) -> Union[Tensor, ndarray]
        """Runs `OverlappingPatches.get` sequentially for each channel.

        :param discard_empty: see `OverlappingPatches.get` docs
        :param concatenate: see `OverlappingPatches.get` docs
        :return: Image patches tensor. Shape is (no_channels * no_pixels_per_patch, no_patches)
                 if concatenate is True else (no_pixels_per_patch, no_patches, no_channels)
        """
        p = [self.OVPs[ch].get(discard_empty) for ch in range(self.no_channels)]
        return (
            (to.cat(p, dim=0) if self._torch else np.concatenate(p, axis=0))
            if concatenate
            else (to.stack(p, dim=-1) if self._torch else np.stack(p, axis=-1))
        )

    def set(
        self,
        new_patches,
        discarded_empty=False,
        concatenated=True,
    ):
        # type: (Union[Tensor, ndarray], bool, bool) -> None
        """Runs `OverlappingPatches.set` sequentially for each channel.

        :param new_patches: Image patches tensor filled with new values (cf.
                            `OverlappingPatches.set` docs). Shape is assumed to be (no_channels *
                            no_pixels_per_patch, no_patches) if concatenated==True and
                            (no_pixels_per_patch, no_patches, no_channels) otherwise.
        :param discarded_empty: see `OverlappingPatches.set` docs
        :param concatenated: Whether patches are concatenated (i.e., whether `get` was called with
                             `concatenate=True`).
        """
        patches = self.get(discarded_empty)
        assert (
            True
            if isinstance(new_patches, ndarray) and isinstance(patches, ndarray)
            else new_patches.device == patches.device  # type: ignore
        ), "device of new_patches mismatched"

        assert new_patches.dtype == patches.dtype, "dtype of new_patches mismatched"
        no_dims = np.ndim(new_patches) if isinstance(new_patches, ndarray) else new_patches.dim()
        to_or_np = to if self._torch else np
        if concatenated:
            assert (
                no_dims == 2
            ), "patches tensor must be two-dimensional\
            (patch_height*patch_width*no_channels x no_patches)"
            px_per_ch = new_patches.shape[0] // self.no_channels
            for ch in range(self.no_channels):
                inds_px_ch = to_or_np.arange(px_per_ch) + ch * px_per_ch
                self.OVPs[ch].set(new_patches[inds_px_ch, :], discarded_empty)
        else:
            assert (
                no_dims == 3
            ), "patches tensor must be three-dimensional (patch_height*patch_width x no_patches\
                x no_channels)"
            for ch in range(self.no_channels):
                self.OVPs[ch].set(new_patches[:, :, ch], discarded_empty)

    def merge(self, merge_method=mean_merger):
        # type: (Callable) -> Union[Tensor, ndarray]
        """Runs `OverlappingPatches.merge` sequentially for each channel

        :param merge_method: See `OverlappingPatches.merge` docs
        :return: Image obtained through patch averaging, is (height, width, no_channels)
        """
        to_stack = [self.OVPs[ch].merge(merge_method) for ch in range(self.no_channels)]
        return to.stack(to_stack, dim=-1) if self._torch else np.stack(to_stack, axis=-1)

    def set_and_merge(
        self,
        new_patches,
        discarded_empty=False,
        concatenated=True,
        merge_method=mean_merger,
    ):
        # type: (Union[Tensor, ndarray], bool, bool, Callable) -> Union[Tensor, ndarray]
        """Runs `OverlappingPatches.set_and_merge` sequentially for each channel.

        :param new_patches: See `set` docs.
        :param discarded_empty: See `set` docs.
        :param concatenated: See `set` docs.
        :param merge_method: See `merge` docs.
        :return: See `merge` docs.
        """
        patches = self.get(discarded_empty)
        if not isinstance(new_patches, ndarray) and not isinstance(patches, ndarray):
            assert new_patches.device == patches.device, "device of new_patches mismatched"
        assert new_patches.dtype == patches.dtype, "dtype of new_patches mismatched"
        no_dims = np.ndim(new_patches) if isinstance(new_patches, ndarray) else new_patches.dim()
        to_or_np = to if self._torch else np
        if concatenated:
            assert (
                no_dims == 2
            ), "patches tensor must be two-dimensional\
            (patch_height*patch_width*no_channels x no_patches)"
            px_per_ch = new_patches.shape[0] // self.no_channels
            merged = [
                self.OVPs[ch].set_and_merge(
                    new_patches[to_or_np.arange(px_per_ch) + ch * px_per_ch, :],
                    discarded_empty,
                    merge_method,
                )
                for ch in range(self.no_channels)
            ]
        else:
            assert (
                no_dims == 3
            ), "patches tensor must be three-dimensional (patch_height*patch_width x no_patches\
                x no_channels)"
            merged = [
                self.OVPs[ch].set_and_merge(new_patches[:, :, ch], discarded_empty, merge_method)
                for ch in range(self.no_channels)
            ]
        return to.stack(merged, dim=-1) if self._torch else np.stack(merged, axis=-1)

class Overlapping3DPatches:
    def __init__(
        self,
        image,
        patch_height,
        patch_width,
        patch_length,
        patch_shift,
        verbose=False,
    ):
        # type: (Union[Tensor, ndarray], int, int, int, bool) -> None
        """Back and forth transformation for image segmentation into overlapping patches (3D).
        Makes use of `skimage.util.view_as_windows`.

        :param image: Tensor to be cut into patches and reconstructed. Must be 3-dimensional,
                      (height, width, length).
        :param patch_height: Will be passed as `window_shape[0]` to `skimage.util.view_as_windows`.
        :param patch_width: Will be passed as `window_shape[1]` to `skimage.util.view_as_windows`.
        :param patch_length: Will be passed as `window_shape[2]` to `skimage.util.view_as_windows`.
        :param patch_shift: Will be passed as `step` to `skimage.util.view_as_windows`.
        :param verbose: Whether to print details when merging patches
        """
        assert np.ndim(image) == 3, "image tensor must be two-dimensional (width x height)"

        self._torch = False if isinstance(image, ndarray) else True
        self._verbose = verbose
        self._patch_height = patch_height
        self._patch_width = patch_width
        self._patch_length = patch_length
        self._patch_shift = patch_shift
        device, precision = None if isinstance(image, ndarray) else image.device, image.dtype
        image_np = image if isinstance(image, ndarray) else image.detach().cpu().numpy()

        # infer some parameters
        image_not_incomplete = np.logical_not(np.isnan(image_np).any())
        image_height, image_width, image_length = image_np.shape[0], image_np.shape[1], image_np.shape[2]
        no_pixels_in_patch = patch_height * patch_width * patch_length
        no_patches_vert = int(
            np.ceil(float(image_height - patch_height) / patch_shift) + 1
        )  # no patches in vertical direction
        no_patches_horz = int(
            np.ceil(float(image_width - patch_width) / patch_shift) + 1
        )  # no patches in horizontal direction
        no_patches_lat = int(
            np.ceil(float(image_length - patch_length) / patch_shift) + 1
        )  # no patches in lateral direction
        no_patches = no_patches_vert * no_patches_horz * no_patches_lat
        no_patches_vert_shift_1 = int(
            np.ceil(float(image_height - patch_height)) + 1
        )  # no patches in vertical dir for step=1
        no_patches_horz_shift_1 = int(
            np.ceil(float(image_width - patch_width)) + 1
        )  # no patches in horizontal dir for step=1
        no_patches_lat_shift_1 = int(
            np.ceil(float(image_length - patch_length)) + 1
        )  # no patches in lateral dir for step=1
        no_patches_shift_1 = (
            no_patches_vert_shift_1 * no_patches_horz_shift_1 * no_patches_lat_shift_1
        )  # no patches for step=1

        ninds_ = np.arange(no_patches_shift_1).reshape(
            no_patches_vert_shift_1, no_patches_horz_shift_1, no_patches_lat_shift_1
        )  # lower right patch locations in image for step = 1
        hinds = (
            np.unique(
                np.append(
                    np.arange(1, no_patches_vert_shift_1, patch_shift),
                    [no_patches_vert_shift_1],
                )
            )
            - 1
        ).flatten()  # indices of relevant patches for step=patch_shift in vertical direction
        winds = (
            np.unique(
                np.append(
                    np.arange(1, no_patches_horz_shift_1, patch_shift),
                    [no_patches_horz_shift_1],
                )
            )
            - 1
        ).flatten()  # indices of relevant patches for step=patch_shift in horizontal direction
        linds = (
            np.unique(
                np.append(
                    np.arange(1, no_patches_lat_shift_1, patch_shift),
                    [no_patches_lat_shift_1],
                )
            )
            - 1
        ).flatten()  # indices of relevant patches for step=patch_shift in lateral direction
        ninds = (ninds_[hinds, :, :][:, winds, :][:, :, linds]).flatten()  # indices of relevant patches
        assert len(ninds) == no_patches
        dinds = np.arange(no_pixels_in_patch).reshape(
            patch_height, patch_width, patch_length
        )  # spatial order of pixel indices, is (patch_height, patch_width),
        # indexed l->r and then top->bottom
        to_be_synthesized = (
            np.isnan(image_np) if np.isnan(image_np).any() else np.ones_like(image_np, dtype=bool)
        )  # indicates which pixels of the input image are to be reconstructed
        ind_rows_to_synthesize, ind_cols_to_synthesize, ind_deps_to_synthesize = np.where(
            to_be_synthesized
        )  # index tuples of missing values, is (total # missing vals)
        no_pixels_to_synthesize = ind_rows_to_synthesize.size  # no missing values

        # cut patches
        print("Extracting patches...", end="")
        patches_np = view_as_windows(
            image_np, window_shape=[patch_height, patch_width, patch_length], step=1
        )  # moves sliding window left->right and then top->bottom
        # is (image_height-patch_height+1, image_width-patch_width+1, patch_height, patch_width)
        patches_np = (
            patches_np.reshape(patch_height, patch_width, patch_length, no_patches_shift_1)
            .reshape(no_patches_shift_1, no_pixels_in_patch)
            .T
        )  # is (no_pixels_in_patch,no_patches_shift_1)
        patches_np = patches_np[
            :, ninds
        ]  # remove patches not satisfying `patch_shift` is (no_pixels_in_patch,no_patches)
        patches_np_not_isnan = np.logical_not(np.isnan(patches_np))
        patches = (
            to.from_numpy(patches_np).to(dtype=precision, device=device)
            if self._torch
            else patches_np
        )
        print("Done")

        # compute indices required to merge patches back to image
        print("Initialize back-transformation...", end="")
        all_inds_relevant_patches = [0] * no_pixels_to_synthesize
        all_inds_relevant_values_in_patch = [0] * no_pixels_to_synthesize
        restorable = (
            np.zeros(no_pixels_to_synthesize, dtype=bool) if np.isnan(patches_np).any() else None
        )
        for p in range(no_pixels_to_synthesize):

            # location of missing value in original image
            r, c, d = ind_rows_to_synthesize[p], ind_cols_to_synthesize[p], ind_deps_to_synthesize[p]

            # location of relevant patches for patch_shift = 1(rows and columns of ninds_)
            r_ = (
                np.arange(
                    max(r - patch_height + 2, 1),
                    min(r + 1, no_patches_vert_shift_1) + 1,
                )
                - 1
            )
            c_ = np.arange(max(c - patch_width + 2, 1), min(c + 1, no_patches_horz_shift_1) + 1) - 1
            d_ = np.arange(max(d - patch_length + 2, 1), min(d + 1, no_patches_lat_shift_1) + 1) - 1

            ns_ = ninds_[r_, :, :][:, c_, :][
                :, :, d_
            ].flatten()  # is no relevant patches for given pixel in original
            ds_ = np.sort(dinds[r - r_, :, :][:, c - c_, :][:, :, d - d_].flatten())[::-1]

            # only use patches compatible with given patch_shift
            if patch_shift > 1:
                nsinds = np.isin(ns_, ninds)
                inds_relevant_patches, inds_relevant_values_in_patch = (
                    ns_[nsinds],
                    ds_[nsinds],
                )
            else:
                inds_relevant_patches, inds_relevant_values_in_patch = ns_, ds_

            # indices considering remaining patches
            if patch_shift > 1:
                inds_relevant_patches = np.where(np.isin(ninds, inds_relevant_patches))[0]

            if image_not_incomplete:
                all_inds_relevant_patches[p] = inds_relevant_patches
                all_inds_relevant_values_in_patch[p] = inds_relevant_values_in_patch
            else:
                relevant_patches = patches_np_not_isnan[:, inds_relevant_patches]
                ind_nonempty_patches = relevant_patches.any(axis=0)
                if ind_nonempty_patches.any():
                    assert restorable is not None  # to make mypy happy
                    restorable[p] = True
                    # ind_nonempty_patches = relevant_patches.any(axis=0)
                    all_inds_relevant_patches[p] = inds_relevant_patches[ind_nonempty_patches]
                    all_inds_relevant_values_in_patch[p] = inds_relevant_values_in_patch[
                        ind_nonempty_patches
                    ]
        print("Done")

        self._image, self._patches = image, patches
        self._ind_rows_to_synthesize = (
            to.from_numpy(ind_rows_to_synthesize).to(dtype=to.int64, device=device)
            if self._torch
            else ind_rows_to_synthesize
        )
        self._ind_cols_to_synthesize = (
            to.from_numpy(ind_cols_to_synthesize).to(dtype=to.int64, device=device)
            if self._torch
            else ind_cols_to_synthesize
        )
        self._ind_deps_to_synthesize = (
            to.from_numpy(ind_deps_to_synthesize).to(dtype=to.int64, device=device)
            if self._torch
            else ind_deps_to_synthesize
        )
        self._no_pixels_to_synthesize = no_pixels_to_synthesize
        self._restorable = (
            (
                to.from_numpy(restorable).to(dtype=to.bool, device=device)
                if self._torch
                else restorable
            )
            if restorable is not None
            else None
        )

        all_inds_relevant_patches = [
            x.copy() if isinstance(x, np.ndarray) else np.ndarray(x)  # type: ignore
            for x in all_inds_relevant_patches
        ]
        self._all_inds_relevant_patches = [
            to.from_numpy(x).to(dtype=to.int64, device=device) if self._torch else x
            for x in all_inds_relevant_patches
        ]

        all_inds_relevant_values_in_patch = [
            x.copy() if isinstance(x, np.ndarray) else np.ndarray(x)  # type: ignore
            for x in all_inds_relevant_values_in_patch
        ]
        self._all_inds_relevant_values_in_patch = [
            to.from_numpy(x).to(dtype=to.int64, device=device) if self._torch else x
            for x in all_inds_relevant_values_in_patch
        ]

    def get_image_shape(self):
        # type: () -> Tuple[int, int]
        """Return shape of input image

        :return: Image shape, (height, width, length)
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
            not_isnan.any(**{"dim" if self._torch else "axis": 0})
        ).item()
        no_patches_without_discarding = int(self._patches.shape[1])
        return no_patches_with_discarding if discard_empty else no_patches_without_discarding

    def get_patch_height_width_shift(self):
        # type: () -> Tuple[int, int, int]
        """Return the patch height, width and shift

        :return: Tuple with (patch height, width, length and shift)
        """
        return self._patch_height, self._patch_width, self._patch_shift, self._patch_length

    def get(self, discard_empty=True):
        # type: (bool) -> Union[Tensor, ndarray]
        """Returns patches cut from image.

        :param discard_empty: Whether to discard patches that do not contain finite entries
        :return: Image patches tensor, is (no_pixels_per_patch, no_patches)
        """
        to_or_np = to if self._torch else np
        if to_or_np.logical_not(to_or_np.isnan(self._patches).any()):
            return self._patches
        else:
            if discard_empty:
                not_isnan = to_or_np.logical_not(to_or_np.isnan(self._patches))  # type: ignore
                inds_not_empty = not_isnan.any(**{"dim" if self._torch else "axis": 0})
                return self._patches[:, inds_not_empty]
            else:
                return self._patches

    def set(self, new_patches, discarded_empty=True):
        # type: (Union[Tensor, ndarray], bool) -> None
        """Update image patches tensor to new values

        :param new_patches: Image patches tensor filled with new values. `self._patches` will
                            be updated to this tensor, must be (no_pixels_per_patch, no_patches).
        :param discarded_empty: Whether patches without finite entries have been discarded when
                                `get` was called (compare docs of `get`).
        """
        to_or_np = to if self._torch else np
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

    def merge(self, merge_method=mean_merger):
        # type: (Callable) -> Union[Tensor, ndarray]
        """Merge patches to obtain new image.

        :param merge_method: Function defining how pixel estimates from different patches are to be
                             merged, defaults to unweighted averaging.
        :return: Image obtained through patch averaging, is (height, width)
        """
        new_image = self._image.copy() if isinstance(self._image, ndarray) else self._image.clone()
        for p in range(self._no_pixels_to_synthesize):
            if self._restorable is not None and not self._restorable[p]:
                continue

            r, c, d = self._ind_rows_to_synthesize[p], self._ind_cols_to_synthesize[p], self._ind_deps_to_synthesize[p]
            inds_relevant_patches = self._all_inds_relevant_patches[p]
            inds_relevant_values_in_patch = self._all_inds_relevant_values_in_patch[p]

            restored = self._patches[inds_relevant_values_in_patch, inds_relevant_patches]

            if self._verbose:
                print("Processing image pixel at ({},{},{})".format(r, c, d))
                print("=" * 12)
                print("Estimates from all patches \n  {}\n".format(restored))

            kwargs = (
                {
                    "height": self._patch_height,
                    "width": self._patch_width,
                    "length": self._patch_length,
                    "inds_relevant": inds_relevant_values_in_patch,
                }
                if merge_method == weighted_mean_merger
                else {}
            )
            estimate = merge_method(restored, **kwargs)

            if self._verbose:
                print("Merged estimate is \n  {}\n".format(estimate))

            new_image[r, c, d] = estimate

        return new_image

    def set_and_merge(
        self,
        new_patches,
        discarded_empty=True,
        merge_method=mean_merger,
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
