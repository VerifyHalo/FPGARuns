import sys
import re
import glob
import io
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "load_intan_rhd_format"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_intan_rhd_format import read_data

# TODO: Replace Input/Output Folders
SRC_DIR = "/Users/anton/Desktop/Day-0_3 KHz"
OUT_DIR = "/Users/anton/Desktop/plots"
N_BINS  = 6000 # downsample resolution
# since we compress 1 hour files

def plot_file(path: str, out_dir: str):
    # file name output
    hour = int(re.search(r"_(\d+) hr", path).group(1))

    buf = io.StringIO()
    # parse header + read every data block
    with contextlib.redirect_stdout(buf):
        d = read_data(path)

    # print(d.keys())
    #   spike_triggers, amplifier_channels, notes, frequency_parameters,
    #   reference_channel, t_amplifier, amplifier_data

    amp = d["amplifier_data"]   # (n_channels, n_samples), microvolts
    t   = d["t_amplifier"]      # seconds / timestampt

    # To Explain (Hitten, Mention Error)
    n_ch, n = amp.shape         # channel count, total samples per channel

    # Official loader reading each RHD block:
    # channel-major (all of ch0's samples, then all of ch1's, ...).
    #
    # FPGARuns' old rhd_reader.py got this backwards: it assumed each block
    # was SAMPLE-major (s0ch0, s0ch1, ..., s0chN, s1ch0, s1ch1, ...).
    # 
    # I was pulling out "channel i" by looping over samples FIRST and picking
    # out every Nth value from inside each sample's mini-group of channels.

    # Visualization Purposes Only
    bin_size = n // N_BINS
    trim = N_BINS * bin_size
    t_plot = t[:trim].reshape(N_BINS, bin_size)[:, 0]

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 9), sharex=True)
    for i in range(n_ch):
        a = amp[i, :trim].reshape(N_BINS, bin_size)
        lo = a.min(axis=1) # scale Y
        hi = a.max(axis=1) # scale Y
        axes[i].fill_between(t_plot, lo, hi, color="#111", linewidth=0)
        axes[i].set_ylabel(f"CH{i}", fontsize=8, rotation=0, labelpad=20, va="center")
        axes[i].tick_params(labelsize=8)
    axes[-1].set_xlabel("s")
    fig.text(0.01, 0.5, "uV", rotation=90, va="center", fontsize=9)
    fig.tight_layout(rect=[0.02, 0, 1, 1])

    out_path = f"{out_dir}/hour{hour:02d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_file_old(path: str, out_dir: str):
    from rhd_reader import _parse_header

    hour = int(re.search(r"_(\d+) hr", path).group(1))

    with open(path, "rb") as f:
        h = _parse_header(f)
        num_amp        = h["num_amp"]
        n_samp         = h["n_samp"]
        bpb            = h["bytes_per_block"]
        skip_after_amp = h["skip_after_amp"]
        header_end     = h["header_end"]
        sample_rate    = h["sample_rate"]

        f.seek(0, 2)
        num_blocks = (f.tell() - header_end) // bpb
        n = num_blocks * n_samp

        amp_wrong = np.empty((num_amp, n), dtype=np.uint16)
        amp_block_bytes = n_samp * num_amp * 2

        f.seek(header_end)
        for b in range(num_blocks):
            f.seek(n_samp * 4, 1)
            flat = np.frombuffer(f.read(amp_block_bytes), dtype="<u2")
            for s in range(n_samp):
                for ch in range(num_amp):
                    amp_wrong[ch, b * n_samp + s] = flat[s * num_amp + ch]
            f.seek(skip_after_amp, 1)

    amp_wrong_uv = (amp_wrong.astype(np.float32) - 32768.0) * 0.195
    t = np.arange(n) / sample_rate

    n_ch = num_amp
    bin_size = n // N_BINS
    trim = N_BINS * bin_size
    t_plot = t[:trim].reshape(N_BINS, bin_size)[:, 0]

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 9), sharex=True)
    for i in range(n_ch):
        a = amp_wrong_uv[i, :trim].reshape(N_BINS, bin_size)
        lo = a.min(axis=1)
        hi = a.max(axis=1)
        axes[i].fill_between(t_plot, lo, hi, color="#c0392b", linewidth=0)  # red = "this is the wrong one"
        axes[i].set_ylabel(f"CH{i}", fontsize=8, rotation=0, labelpad=20, va="center")
        axes[i].tick_params(labelsize=8)
    axes[-1].set_xlabel("s")
    fig.text(0.01, 0.5, "uV", rotation=90, va="center", fontsize=9)
    fig.tight_layout(rect=[0.02, 0, 1, 1])

    out_path = f"{out_dir}/hour{hour:02d}_OLD_WRONG.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    files = sorted(
        glob.glob(f"{SRC_DIR}/*.rhd"),
        key=lambda p: int(re.search(r"_(\d+) hr", p).group(1)),
    )
    for path in files:
        plot_file(path, OUT_DIR)

if __name__ == "__main__":
    main()
