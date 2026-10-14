"""Detection of disconnected channels."""

import numpy as np


def find_disconnected(raw, processed, flat_fraction=0.9, min_kurtosis=5.0):
    """Return a bool array, True for channels that are not connected.

    Two symptoms are checked:
    - flat line (zeros, a constant, or stuck at an ADC rail): most consecutive
      samples are identical in the raw signal;
    - a floating electrode: after filtering only noise is left, and noise has
      a kurtosis near 3 while an ECG, dominated by sharp QRS complexes, is
      well above 5 (the kSQI index of Li, Mark and Clifford, 2008).
    """
    raw, processed = np.atleast_2d(raw), np.atleast_2d(processed)
    flat = np.mean(np.diff(raw, axis=1) == 0, axis=1) > flat_fraction
    return flat | (kurtosis(processed) < min_kurtosis)


def kurtosis(x):
    """Kurtosis of each row (3 for a Gaussian); 0 for a constant row."""
    x = x - x.mean(axis=1, keepdims=True)
    m2, m4 = np.mean(x**2, axis=1), np.mean(x**4, axis=1)
    return np.divide(m4, m2**2, out=np.zeros_like(m2), where=m2 > 0)
