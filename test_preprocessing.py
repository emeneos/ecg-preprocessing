"""Verification of the pre-processing: data types, expected results per step,
disconnected channels, the whole pipeline (synthetic and provided data) and
computational time. Run with `python -m pytest`.
"""

import time
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import preprocessing as pp
from lead_off import find_disconnected
from mat_io import load_ecg

FS = 1000.0
EXAMPLE = Path(__file__).parent / "data" / "ecgConditioningExample.mat"


# --- synthetic signals with known content -----------------------------------

def clean_ecg(duration_s=20.0, bpm=70.0, seed=0):
    """Sum of Gaussian P, Q, R, S and T waves per beat; R wave height is 1."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * FS)) / FS
    waves = [(-0.20, 0.025, 0.12), (-0.035, 0.010, -0.1), (0.0, 0.012, 1.0),
             (0.035, 0.010, -0.2), (0.28, 0.050, 0.3)]
    ecg = np.zeros_like(t)
    beat = 0.3
    while beat < duration_s + 0.5:
        for offset, width, amp in waves:
            ecg += amp * np.exp(-0.5 * ((t - beat - offset) / width) ** 2)
        beat += 60.0 / bpm * (1 + 0.03 * rng.standard_normal())
    return ecg


def mains(n, hz=50.0, amp=0.3):
    t = np.arange(n) / FS
    return amp * (np.sin(2 * np.pi * hz * t) + 0.5 * np.sin(2 * np.pi * 2 * hz * t + 1))


def noisy_ecg():
    """Clean ECG plus mains, baseline wander, offset and two spikes."""
    ecg = clean_ecg()
    t = np.arange(ecg.size) / FS
    x = ecg + mains(ecg.size) + 0.5 * np.sin(2 * np.pi * 0.15 * t) + 2000.0
    x[5000:5004] += 1e4
    x[12000:12004] -= 1e4
    return x, ecg


def gain_db(filter_fn, hz, duration_s=20.0):
    """Gain of a filter at one frequency, measured on a sine (edges excluded)."""
    t = np.arange(int(duration_s * FS)) / FS
    x = np.sin(2 * np.pi * hz * t)
    y = filter_fn(x[np.newaxis, :])[0]
    core = slice(int(3 * FS), -int(3 * FS))
    return 20 * np.log10(np.std(y[core]) / np.std(x[core]))


# --- data types -------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32])
def test_output_is_float64_for_any_numeric_input(dtype):
    x = (1000 * clean_ecg(5.0)).astype(dtype)
    result = pp.preprocess(x, FS)
    assert result.signals.dtype == np.float64
    assert pp.lowpass(x, FS).dtype == np.float64


def test_shape_is_preserved():
    x = np.vstack([clean_ecg(5.0), clean_ecg(5.0, bpm=90)])
    assert pp.preprocess(x, FS).signals.shape == x.shape
    assert pp.preprocess(x[0], FS).signals.shape == (1, x.shape[1])


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        pp.preprocess(np.zeros((2, 3, 100)), FS)
    with pytest.raises(ValueError):
        pp.preprocess(np.array(["a", "b"]), FS)


# --- expected results of each step ------------------------------------------

def test_no_false_spikes_on_clean_ecg():
    for bpm in (40, 70, 150):
        assert not pp.find_spikes(clean_ecg(bpm=bpm), FS).any()


def test_spikes_are_removed_where_they_are():
    x, ecg = noisy_ecg()
    y, mask = pp.remove_spikes(x[np.newaxis, :], FS, powerline_hz=50.0)
    assert mask[0, 5000:5004].all() and mask[0, 12000:12004].all()
    assert mask.sum() == 2 * (4 + 2 * 2)  # the spikes plus a 2 ms margin each side
    assert np.abs(y - x).max() > 9e3 and np.abs(y[0] - 2000).max() < 5  # spikes gone, ECG kept


def test_offset_is_removed():
    y = pp.remove_offset(clean_ecg() + 1.675e7)
    assert abs(np.median(y)) < 1e-6


def test_notch_removes_mains_and_keeps_ecg_band():
    notch = lambda x: pp.remove_powerline(x, FS, 50.0)
    for hz in (50, 100, 150):
        assert gain_db(notch, hz) < -40
    assert abs(gain_db(notch, 10)) < 0.1


def test_lowpass_response():
    lp = lambda x: pp.lowpass(x, FS, 150.0)
    assert abs(gain_db(lp, 10)) < 0.1
    assert gain_db(lp, 300) < -40


def test_highpass_response():
    hp = lambda x: pp.remove_baseline(x, FS, 0.5)
    assert gain_db(hp, 0.1, duration_s=60.0) < -20
    assert abs(gain_db(hp, 5)) < 0.1


def test_r_peaks_do_not_move():
    x, ecg = noisy_ecg()
    y = pp.preprocess(x, FS).signals[0]
    core = slice(int(FS), -int(FS))  # the first and last second carry edge effects
    peaks_in, _ = signal.find_peaks(ecg[core], height=0.5, distance=300)
    peaks_out, _ = signal.find_peaks(y[core], height=0.5, distance=300)
    assert len(peaks_in) == len(peaks_out)
    assert np.abs(peaks_out - peaks_in).max() <= 1


# --- disconnected channels --------------------------------------------------

def good_channels():
    """Four valid ECGs with different gain, offset and polarity, plus mains."""
    ecg = clean_ecg()
    rows = [1e4 * g * ecg + 1e4 * mains(ecg.size) + off
            for g, off in ((1.0, 3e4), (0.7, 1.675e7), (-0.8, 2e4), (1.3, 0))]
    return np.vstack(rows)


@pytest.mark.parametrize("bad", [
    np.zeros(20000),
    np.full(20000, 1234.0),
    3e4 + 5e3 * np.random.default_rng(0).standard_normal(20000),
    3e4 + 1e4 * mains(20000) + 2e3 * np.random.default_rng(1).standard_normal(20000),
])
def test_flat_and_noise_channels_are_flagged(bad):
    result = pp.preprocess(np.vstack([good_channels(), bad]), FS)
    assert result.disconnected.tolist() == [False] * 4 + [True]
    assert np.isnan(result.signals[-1]).all()


def test_good_channels_are_not_flagged():
    assert not pp.preprocess(good_channels(), FS).disconnected.any()


# --- whole pipeline ---------------------------------------------------------

def test_pipeline_recovers_clean_ecg():
    x, ecg = noisy_ecg()
    y = pp.preprocess(x, FS).signals[0]
    # The clean ECG goes through the same band so edges and DC compare alike.
    ref = pp.remove_baseline(pp.lowpass(ecg, FS), FS)[0]
    core = slice(int(1 * FS), -int(1 * FS))
    assert np.corrcoef(y[core], ref[core])[0, 1] > 0.995
    assert np.sqrt(np.mean((y[core] - ref[core]) ** 2)) < 0.01 * np.ptp(ref)


def test_example_file_channels_and_spikes():
    ecg, fs = load_ecg(EXAMPLE)
    assert ecg.shape == (6, 30000) and fs == 1000.0
    result = pp.preprocess(ecg, fs)
    assert result.disconnected.tolist() == [False, False, False, True, False, False]
    assert np.isfinite(result.signals[~result.disconnected]).all()
    assert result.spike_mask[2, 9999:10003].all()
    assert result.spike_mask[5, 9999:10003].all() and result.spike_mask[5, 24999:25003].all()
    assert np.abs(result.signals[[2, 5]]).max() < 5e4  # the raw spikes reached 1.3e8


def test_example_file_mains_removed():
    ecg, fs = load_ecg(EXAMPLE)
    y = pp.preprocess(ecg, fs).signals[1]
    f, p_raw = signal.welch(ecg[1] - np.median(ecg[1]), fs, nperseg=4096)
    _, p_out = signal.welch(y, fs, nperseg=4096)
    for hz in (50, 100, 150):
        band = np.abs(f - hz) < 1
        assert 10 * np.log10(p_raw[band].sum() / p_out[band].sum()) > 30
    ecg_band = (f > 2) & (f < 40)
    assert abs(10 * np.log10(p_out[ecg_band].sum() / p_raw[ecg_band].sum())) < 0.5


# --- computational time -----------------------------------------------------

def test_processing_time():
    ecg, fs = load_ecg(EXAMPLE)
    pp.preprocess(ecg, fs)  # warm-up
    steps = [
        ("estimate_powerline_hz", lambda: pp.estimate_powerline_hz(ecg, fs)),
        ("remove_spikes", lambda: pp.remove_spikes(ecg, fs, 50.0)),
        ("remove_offset", lambda: pp.remove_offset(ecg)),
        ("remove_powerline", lambda: pp.remove_powerline(ecg, fs, 50.0)),
        ("lowpass", lambda: pp.lowpass(ecg, fs)),
        ("remove_baseline", lambda: pp.remove_baseline(ecg, fs)),
        ("find_disconnected", lambda: find_disconnected(ecg, ecg)),
        ("preprocess (total)", lambda: pp.preprocess(ecg, fs)),
    ]
    print()
    for name, fn in steps:
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        print(f"  {name:<22}{1e3 * elapsed:7.1f} ms")
        if name.startswith("preprocess"):
            assert elapsed < 1.0
