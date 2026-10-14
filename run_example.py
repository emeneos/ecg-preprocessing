"""Pre-process the provided recording and save the result and two figures.

    python run_example.py [file.mat]
"""

import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from mat_io import load_ecg, save_ecg
from preprocessing import preprocess

path = sys.argv[1] if len(sys.argv) > 1 else "data/ecgConditioningExample.mat"
ecg, fs = load_ecg(path)

start = time.perf_counter()
result = preprocess(ecg, fs)
print(f"{ecg.shape[0]} channels, {ecg.shape[1] / fs:g} s at {fs:g} Hz, "
      f"processed in {1e3 * (time.perf_counter() - start):.0f} ms")
print(f"powerline frequency: {result.powerline_hz:g} Hz")
for ch in range(ecg.shape[0]):
    status = "disconnected" if result.disconnected[ch] else "ok"
    spikes = int(result.spike_mask[ch].sum())
    print(f"channel {ch}: {status}" + (f", {spikes} samples replaced" if spikes else ""))

save_ecg("ecg_preprocessed.mat", result.signals, fs,
         disconnected=result.disconnected.astype(np.uint8))

# Figure 1: every channel, raw and processed, plus a 3 s zoom.
t = np.arange(ecg.shape[1]) / fs
zoom = slice(int(5 * fs), int(8 * fs))
fig, axes = plt.subplots(ecg.shape[0], 3, figsize=(16, 2 * ecg.shape[0]))
for ch, (a_raw, a_out, a_zoom) in enumerate(axes):
    a_raw.plot(t, ecg[ch], lw=0.4, color="0.3")
    a_out.plot(t, result.signals[ch], lw=0.4)
    a_zoom.plot(t[zoom], ecg[ch, zoom] - np.median(ecg[ch]), lw=0.5, color="0.7")
    a_zoom.plot(t[zoom], result.signals[ch, zoom], lw=0.8)
    a_raw.set_ylabel(f"ch {ch}")
    if result.disconnected[ch]:
        a_out.set_title("disconnected", color="C3", fontsize=9)
axes[0, 0].set_title("raw (ADC counts)")
axes[0, 1].set_title("pre-processed")
axes[0, 2].set_title("5-8 s, raw minus offset vs pre-processed")
fig.tight_layout()
fig.savefig("figure_signals.png", dpi=100)

# Figure 2: spectrum of one connected channel before and after.
ch = int(np.flatnonzero(~result.disconnected)[0])
f, p_raw = signal.welch(ecg[ch] - np.median(ecg[ch]), fs, nperseg=int(4 * fs))
_, p_out = signal.welch(result.signals[ch], fs, nperseg=int(4 * fs))
plt.figure(figsize=(10, 4))
plt.semilogy(f, p_raw, lw=0.8, color="0.5", label="raw")
plt.semilogy(f, p_out, lw=0.8, label="pre-processed")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title(f"channel {ch}")
plt.legend()
plt.tight_layout()
plt.savefig("figure_spectrum.png", dpi=100)
print("saved ecg_preprocessed.mat, figure_signals.png, figure_spectrum.png")
