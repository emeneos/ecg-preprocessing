"""Reading and writing the ECG as MATLAB .mat files.

MATLAB stores the signal as (n_samples, n_channels); this package works with
(n_channels, n_samples), so the orientation is converted here and only here.
"""

import numpy as np
import scipy.io


def load_ecg(path):
    """Return (ecg, fs) with ecg shaped (n_channels, n_samples)."""
    version, _ = scipy.io.matlab.matfile_version(path)
    if version == 2:  # v7.3 files are HDF5 and need h5py
        import h5py
        with h5py.File(path, "r") as f:
            ecg = np.array(f["ecg"])  # HDF5 is row-major: already (channels, samples)
            fs = float(np.array(f["fs"]).squeeze())
    else:
        content = scipy.io.loadmat(path)
        ecg = content["ecg"].T
        fs = float(content["fs"].squeeze())
    return np.ascontiguousarray(ecg, dtype=np.float64), fs


def save_ecg(path, ecg, fs, **extra):
    """Save in MATLAB orientation (n_samples, n_channels), v5 format."""
    scipy.io.savemat(path, {"ecg": np.asarray(ecg).T, "fs": fs, **extra})
