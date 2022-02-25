# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
from scipy.linalg import eigh


def whiten(X: np.ndarray, var=0.95, bias=0.1, precision=np.float64) -> np.ndarray:
    """Zero-Phase Component Whitening (compare [1,2])

    :param X: Data, is (n_samples, n_features)
    :param var: Variance to retain
    :param bias: Add this constant to the data variances before computing eigenvectors/-values
    :param precision: Cast data to this floating point precision
    :return: Whitened data

    Adapted based on implementation by Georgios Exarchakis

    [1] A. J. Bell and T. J. Sejnowski. The “independent components” of natural scenes are edge
    filters. Vision Research, 37(23):3327–38, 1997.
    [2] G. Exarchakis and J. Lücke. Discrete Sparse Coding. Neural Computation, 29:2979–3013,
    2017.
    """
    assert np.ndim(X) == 2, "Input must be 2-dim."
    X = X.astype(precision)
    n_samples, n_features = X.shape

    mean_ = np.mean(X, axis=0)
    X -= mean_

    eigs, eigv = eigh(np.dot(X.T, X) / n_samples + bias * np.identity(n_features))
    inds = np.argsort(eigs)[::-1]
    eigs = eigs[inds]
    eigv = eigv[:, inds]
    neigs = eigs / np.sum(eigs)
    nc = np.arange(eigs.shape[0])[np.cumsum(neigs) >= var][0]
    eigs = eigs[:nc]
    eigv = eigv[:, :nc]
    components = np.dot(eigv * np.sqrt(1.0 / eigs), eigv.T)

    X_transformed = np.dot(X, components)
    return X_transformed
