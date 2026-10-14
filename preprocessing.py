"""ECG pre-processing: spikes, offset, powerline, low-pass and baseline wander.

Signals are float64 arrays shaped (n_channels, n_samples). Every function
returns a new array and leaves its input untouched.
"""

from typing import NamedTuple

import numpy as np
from scipy import signal

from lead_off import find_disconnected


class Result(NamedTuple):
    signals: np.ndarray       # (n_channels, n_samples); disconnected rows are NaN
    disconnected: np.ndarray  # bool per channel
    spike_mask: np.ndarray    # bool per sample, True where a spike was replaced
    powerline_hz: float


def as_2d(x):
    """Convert input to a float64 array with one row per channel."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D data, got {x.ndim}-D")
    return x


def preprocess(ecg, fs, lowpass_hz=150.0, highpass_hz=0.5, notch_width_hz=1.0):
    """Run the whole pipeline. See README for the choice of parameters."""
    raw = as_2d(ecg)
    powerline_hz = estimate_powerline_hz(raw, fs)
    # Spikes first: a filter would spread them over hundreds of milliseconds.
    x, spike_mask = remove_spikes(raw, fs, powerline_hz)
    x = remove_offset(x)
    x = remove_powerline(x, fs, powerline_hz, notch_width_hz)
    x = lowpass(x, fs, lowpass_hz)
    x = remove_baseline(x, fs, highpass_hz)

    disconnected = find_disconnected(raw, x)
    x[disconnected] = np.nan
    return Result(x, disconnected, spike_mask, powerline_hz)


# --- spikes -----------------------------------------------------------------

def find_spikes(x, fs, threshold=5.0):
    """Flag samples whose distance to the local baseline is far above the QRS.

    The reference is the 99th percentile of that distance, which sits near the
    top of the R waves; a spike is anything above `threshold` times it.
    """
    x = as_2d(x)
    baseline = block_median(x, int(0.5 * fs))
    distance = np.abs(x - baseline)
    reference = np.percentile(distance, 99, axis=1, keepdims=True)
    return (distance > threshold * reference) & (reference > 0)


def block_median(x, block):
    """Piecewise-constant baseline: the median of each block of samples."""
    n = x.shape[1]
    pad = (-n) % block
    padded = np.pad(x, ((0, 0), (0, pad)), mode="edge")
    medians = np.median(padded.reshape(x.shape[0], -1, block), axis=2)
    return np.repeat(medians, block, axis=1)[:, :n]


def remove_spikes(x, fs, powerline_hz, threshold=5.0, margin_s=0.002):
    """Replace spikes (plus a 2 ms margin for their edges) and return the mask.

    Each gap is filled with a copy of the signal one powerline period earlier.
    A straight line would cut a piece out of the mains sine wave, and the
    notch filter would later turn that missing piece into a bump.
    """
    x = as_2d(x).copy()
    mask = find_spikes(x, fs, threshold)
    margin = int(round(margin_s * fs))
    period = int(round(fs / powerline_hz))
    for ch in np.flatnonzero(mask.any(axis=1)):
        for start, stop in runs(mask[ch]):
            start, stop = max(0, start - margin), min(x.shape[1], stop + margin)
            mask[ch, start:stop] = True
            x[ch, start:stop] = fill_gap(x[ch], start, stop, period)
    return x, mask


def fill_gap(x, start, stop, period):
    """Samples to put in x[start:stop]: one mains period earlier (or later)."""
    length = stop - start
    if start >= period and length <= period:
        return x[start - period:stop - period]
    if stop + period <= x.size and length <= period:
        return x[start + period:stop + period]
    # Long gap or very short signal: a straight line between the neighbours.
    left = x[start - 1] if start > 0 else x[stop]
    right = x[stop] if stop < x.size else x[start - 1]
    return np.linspace(left, right, length)


def runs(mask):
    """(start, stop) index pairs of the True runs in a boolean vector."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return zip(edges[::2], edges[1::2])


# --- offset and powerline ---------------------------------------------------

def remove_offset(x):
    """Subtract the median of each channel (robust to leftover artefacts)."""
    x = as_2d(x)
    return x - np.median(x, axis=1, keepdims=True)


def estimate_powerline_hz(x, fs):
    """Mains frequency (near 50 or 60 Hz) measured on the signal itself."""
    x = as_2d(x)
    # Spikes are broadband and would swamp the spectrum, so clip extremes first.
    lo, hi = np.percentile(x, [0.5, 99.5], axis=1, keepdims=True)
    x = np.clip(x, lo, hi) - np.median(x, axis=1, keepdims=True)
    freqs, power = signal.welch(x, fs, nperseg=min(x.shape[1], int(8 * fs)), axis=1)
    power = power.sum(axis=0)
    best = 0.0
    for nominal in (50.0, 60.0):
        band = (freqs > nominal - 1) & (freqs < nominal + 1)
        peak = freqs[band][np.argmax(power[band])]
        if power[band].max() > best:
            best, powerline_hz = power[band].max(), peak
    return float(powerline_hz)


def remove_powerline(x, fs, powerline_hz, width_hz=1.0):
    """Zero-phase notch at the mains frequency and all its harmonics below Nyquist."""
    x = as_2d(x)
    harmonics = np.arange(powerline_hz, 0.98 * fs / 2, powerline_hz)
    sos = np.vstack([signal.tf2sos(*signal.iirnotch(f, f / width_hz, fs=fs)) for f in harmonics])
    return zero_phase(sos, x, fs)


# --- band limits ------------------------------------------------------------

def lowpass(x, fs, cutoff_hz=150.0, order=4):
    """Zero-phase Butterworth low-pass."""
    sos = signal.butter(order, cutoff_hz, "lowpass", fs=fs, output="sos")
    return zero_phase(sos, as_2d(x), fs)


def remove_baseline(x, fs, cutoff_hz=0.5, order=2):
    """Zero-phase Butterworth high-pass; also removes any remaining offset."""
    sos = signal.butter(order, cutoff_hz, "highpass", fs=fs, output="sos")
    return zero_phase(sos, as_2d(x), fs)


def zero_phase(sos, x, fs):
    """Forward-backward filtering with 3 s of edge padding.

    The default padding of sosfiltfilt is a few samples, shorter than the
    transient of a narrow notch or a 0.5 Hz high-pass (about 0.3 s each).
    """
    padlen = min(x.shape[1] - 1, int(3 * fs))
    return signal.sosfiltfilt(sos, x, axis=1, padlen=padlen)
