import struct
import numpy as np

MAGIC = 0xC6912702


# ── internal ──────────────────────────────────────────────────────────────────

def _read_qstring(f):
    (length,) = struct.unpack("<I", f.read(4))
    if length in (0, 0xFFFFFFFF):
        return ""
    num_chars = length // 2
    chars = struct.unpack(f"<{num_chars}H", f.read(num_chars * 2))
    return "".join(chr(c) if c < 128 else "?" for c in chars)


def _parse_header(f):
    """Parse RHD header from an open file. Returns a dict of header fields."""
    (magic,) = struct.unpack("<I", f.read(4))
    if magic != MAGIC:
        raise ValueError("Not a valid RHD file (bad magic number)")

    ver_major, ver_minor = struct.unpack("<HH", f.read(4))
    (sample_rate,) = struct.unpack("<f", f.read(4))

    f.seek(36, 1)               # skip: freq settings (26) + notch (2) + impedance (8)
    for _ in range(3):          # skip 3 note QStrings
        _read_qstring(f)

    num_temp = 0
    if ver_major > 1 or (ver_major == 1 and ver_minor >= 1):
        (num_temp,) = struct.unpack("<H", f.read(2))
    if ver_major > 1 or (ver_major == 1 and ver_minor >= 3):
        f.seek(2, 1)            # skip eval board mode
    if ver_major > 1:
        _read_qstring(f)        # skip reference channel name

    num_amp = num_aux = num_supply = num_adc = 0
    has_dig_in = has_dig_out = False

    (num_groups,) = struct.unpack("<H", f.read(2))
    for _ in range(num_groups):
        _read_qstring(f)
        _read_qstring(f)
        grp_enabled, grp_num_ch, _dummy = struct.unpack("<HHH", f.read(6))
        if grp_enabled and grp_num_ch:
            for _ in range(grp_num_ch):
                _read_qstring(f)
                _read_qstring(f)
                _n, _c, sig_type, ch_en, _chip, _board = struct.unpack("<hhhhhh", f.read(12))
                f.seek(16, 1)   # skip trigger (8) + impedance (8)
                if ch_en:
                    if   sig_type == 0: num_amp    += 1
                    elif sig_type == 1: num_aux    += 1
                    elif sig_type == 2: num_supply += 1
                    elif sig_type == 3: num_adc    += 1
                    elif sig_type == 4: has_dig_in  = True
                    elif sig_type == 5: has_dig_out = True

    n_samp = 60 if ver_major == 1 else 128

    bytes_per_block = (
        n_samp * 4 +
        n_samp * num_amp * 2 +
        (n_samp // 4) * num_aux * 2 +
        num_supply * 2 +
        n_samp * num_adc * 2 +
        (n_samp * 2 if has_dig_in  else 0) +
        (n_samp * 2 if has_dig_out else 0) +
        num_temp * 2
    )

    # skip order within each block (after amp data), matches rhd_reader.cpp
    skip_after_amp = (
        (n_samp // 4) * num_aux * 2 +
        num_supply * 2 +
        num_temp * 2 +
        n_samp * num_adc * 2 +
        (n_samp * 2 if has_dig_in  else 0) +
        (n_samp * 2 if has_dig_out else 0)
    )

    return dict(
        sample_rate=float(sample_rate),
        num_amp=num_amp,
        n_samp=n_samp,
        bytes_per_block=bytes_per_block,
        skip_after_amp=skip_after_amp,
        header_end=f.tell(),
    )


# ── public API ────────────────────────────────────────────────────────────────

def read_rhd_info(path):
    """Read only the header. Fast — does not touch sample data.

    Returns
    -------
    num_channels : int
    sample_rate  : float  (Hz)
    """
    with open(path, "rb") as f:
        h = _parse_header(f)
    if h["num_amp"] == 0:
        raise ValueError(f"No amplifier channels in {path}")
    return h["num_amp"], h["sample_rate"]


def read_rhd_channel(path, channel_idx):
    """Read one amplifier channel from an RHD2000 file.

    channel_idx : 0-indexed amplifier channel.
                  Note: HALOReader maps FPGA channel N → RHD amp index N+2
                  (indices 0-1 are reference/ground electrodes).

    Returns
    -------
    raw_uint16  : np.ndarray  shape (num_samples,)  dtype uint16
    sample_rate : float  (Hz)
    """
    with open(path, "rb") as f:
        h = _parse_header(f)

        num_amp        = h["num_amp"]
        n_samp         = h["n_samp"]
        bpb            = h["bytes_per_block"]
        skip_after_amp = h["skip_after_amp"]
        header_end     = h["header_end"]
        sample_rate    = h["sample_rate"]

        if num_amp == 0:
            raise ValueError(f"No amplifier channels in {path}")
        if channel_idx >= num_amp:
            raise ValueError(f"Channel {channel_idx} out of range ({num_amp} channels in {path})")

        f.seek(0, 2)
        num_blocks    = (f.tell() - header_end) // bpb
        total_samples = num_blocks * n_samp

        amp_block_bytes = n_samp * num_amp * 2
        out = np.empty(total_samples, dtype=np.uint16)

        f.seek(header_end)
        for b in range(num_blocks):
            f.seek(n_samp * 4, 1)                                          # timestamps
            raw = np.frombuffer(f.read(amp_block_bytes), dtype="<u2").reshape(n_samp, num_amp)
            out[b * n_samp : (b + 1) * n_samp] = raw[:, channel_idx]
            f.seek(skip_after_amp, 1)

    return out, float(sample_rate)
