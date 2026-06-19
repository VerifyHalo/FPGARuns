#!/usr/bin/env python3
"""
BCI Recording Viewer
NOTE: GUI built wtih Claude.
"""

import sys
import re
import json
import math
import threading
from pathlib import Path

from PyQt6.QtCore import QTimer

import numpy as np
import matplotlib
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QComboBox, QTreeView, QSplitter, QFrame, QPushButton,
    QSizePolicy, QLineEdit, QFileDialog,
)
from PyQt6.QtGui import QFileSystemModel, QStandardItemModel, QStandardItem, QIcon
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QDir, QModelIndex

from rhd_reader import read_rhd_info, read_rhd_channel
from neo_detector import NeoDetector

def _resource(rel: str) -> Path:
    """Resolve a bundled resource path (works in dev and PyInstaller onefile)."""
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / rel
    return Path(__file__).parent / rel

def _app_dir() -> Path:
    """Directory next to the running executable (for user data files)."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent

DATA_DIR     = _app_dir() / "data"
ANNOT_FILE   = _app_dir() / "annotations.json"
UV_PER_COUNT = 0.195
MAX_PTS      = 20_000
ZOOM_STEP    = 0.80


def _load_annots() -> dict:
    if ANNOT_FILE.exists():
        try:
            return json.loads(ANNOT_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_annots(data: dict):
    ANNOT_FILE.write_text(json.dumps(data, indent=2))


# ── background loader ─────────────────────────────────────────────────────────

class Loader(QThread):
    done  = pyqtSignal(object, object, float, int, object)  # uv, neo, fs, ch, auto_dets
    error = pyqtSignal(str)

    def __init__(self, path, ch):
        super().__init__()
        self.path = path
        self.ch   = ch

    def run(self):
        try:
            raw, fs = read_rhd_channel(self.path, self.ch)
            uv = (raw.astype(np.float32) - 32_768.0) * UV_PER_COUNT
            x = raw.astype(np.int64) - 32_768
            neo = np.zeros(len(x), dtype=np.int64)
            neo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
            neo_uv2 = (np.abs(neo) * (UV_PER_COUNT ** 2)).astype(np.float32)
            # auto-detect with default FPGA parameters
            det_samples = NeoDetector(fs).detect(raw)
            auto_dets = [(s / fs, e / fs) for s, e in det_samples]
            self.done.emit(uv, neo_uv2, fs, self.ch, auto_dets)
        except Exception as e:
            self.error.emit(str(e))


# ── time ruler helpers ────────────────────────────────────────────────────────

_TIME_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30,
               60, 120, 300, 600, 900, 1800, 3600]


def _fmt_abs(t: int) -> str:
    if t <= 0:
        return "0s"
    if t < 60:
        return f"{t}s"
    m, s = divmod(t, 60)
    if m < 60:
        return f"{m}m {s}s" if s else f"{m}m"
    h, rm = divmod(m, 60)
    return f"{h}h" if rm == 0 else f"{h}h {rm}m"


class _TimeLocator(mticker.Locator):
    _TARGET = 6

    def __init__(self):
        super().__init__()
        self.step = 1.0

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        span = max(vmax - vmin, 1e-9)
        self.step = min(_TIME_STEPS, key=lambda s: abs(span / s - self._TARGET))
        t0 = math.ceil(vmin / self.step) * self.step
        ticks, t = [], t0
        while t <= vmax + self.step * 1e-6:
            ticks.append(round(t, 9))
            t += self.step
        return ticks


def _make_fmt(loc):
    def _fmt(x, _=None):
        x = round(x, 9)
        if loc.step >= 1:
            return _fmt_abs(int(round(x)))
        # sub-second: full label only at whole-second boundaries, offset elsewhere
        whole = int(x)
        sub = round(x - whole, 9)
        if sub < 1e-6:
            return _fmt_abs(whole)
        return f"{sub:.1f}s"
    return _fmt


# ── canvas with zoom/pan ──────────────────────────────────────────────────────

MIN_SPAN = 0.1          # maximum zoom: 100 ms window


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal sliding-window mean. out[i] = mean(x[i-w+1 : i+1]), nan for i < w-1."""
    if w <= 1:
        return x.astype(np.float32)
    n = len(x)
    cs = np.cumsum(x, dtype=np.float64)
    out = np.empty(n, dtype=np.float32)
    out[:w - 1] = np.nan
    out[w - 1] = cs[w - 1] / w
    if n > w:
        out[w:] = (cs[w:] - cs[:n - w]) / w
    return out


def _envelope(x: np.ndarray, i0: int, i1: int, fs: float, n_bins: int):
    """Min/max envelope downsample of x[i0:i1] into up to n_bins bins.

    Plain stride-slicing (skip every Nth sample) can miss a spike entirely
    if it falls between kept samples. Binning and keeping each bin's true
    min/max guarantees transients always show up, same as the static plots
    in plot_all_hours.py. Returns (bin_centers, lo, hi).
    """
    n = i1 - i0
    if n <= 0:
        empty = np.array([])
        return empty, empty, empty
    bin_size = max(1, n // n_bins)
    nb = max(1, n // bin_size)
    end = i0 + nb * bin_size
    seg = x[i0:end].reshape(nb, bin_size)
    lo = seg.min(axis=1)
    hi = seg.max(axis=1)
    centers = (i0 + (np.arange(nb) + 0.5) * bin_size) / fs
    return centers, lo, hi


def _envelope_xy(centers: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Zigzag (lo, hi, lo, hi, ...) trace at each bin center, as ONE
    continuous line for a plain Line2D — fast to render (a single connected
    path), while still showing every bin's true min/max swing.
    """
    n = len(centers)
    t_env = np.repeat(centers, 2)
    y_env = np.empty(2 * n, dtype=np.float64)
    y_env[0::2] = lo
    y_env[1::2] = hi
    return t_env, y_env


def _run_gate(triggered: np.ndarray, tc: int, wt: int) -> list[tuple[int, int]]:
    """Gate state machine: tc consecutive detections → seizure start; wt timeouts → end."""
    seizures: list[tuple[int, int]] = []
    NORMAL, SEIZURE = 0, 1
    state = NORMAL
    det_count = timeout_count = 0
    seizure_start = 0
    for i, det in enumerate(triggered):
        if state == NORMAL:
            if det:
                det_count += 1
                if det_count >= tc:
                    state = SEIZURE
                    seizure_start = max(0, i - (tc - 1))
                    det_count = 0
            else:
                det_count = 0
        else:
            if det:
                timeout_count = 0
            else:
                timeout_count += 1
                if timeout_count >= wt:
                    end = i - wt
                    seizures.append((seizure_start, max(seizure_start, end)))
                    state = NORMAL
                    det_count = timeout_count = 0
    if state == SEIZURE:
        seizures.append((seizure_start, len(triggered) - 1))
    return seizures


class WaveCanvas(FigureCanvas):
    annotation_done = pyqtSignal(float, float)   # t0, t1 in seconds
    gate2_updated   = pyqtSignal(object)          # list of (t0, t1) in seconds

    def __init__(self):
        fig = Figure(facecolor="white")
        super().__init__(fig)
        self._ax, self._ax2, self._ax3, self._ax_gate, self._ax4, self._ax5, self._ax_gate2 = fig.subplots(
            7, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 0.6, 3, 2, 0.6, 3], "hspace": 0.25},
        )
        fig.subplots_adjust(top=0.92, bottom=0.05, left=0.08, right=0.98)

        for ax in (self._ax, self._ax2, self._ax3, self._ax_gate, self._ax4, self._ax5, self._ax_gate2):
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        self._ax.set_ylabel("µV", fontsize=9)
        self._ax2.set_ylabel("|NEO| µV²", fontsize=9)
        self._ax_gate.set_ylabel("µV (gate)", fontsize=9)
        self._ax4.set_ylabel("|NEO avg| µV²", fontsize=9)
        self._ax_gate2.set_ylabel("µV (avg gate)", fontsize=9)

        for strip_ax in (self._ax3, self._ax5):
            strip_ax.set_ylabel("det.", fontsize=8)
            strip_ax.set_yticks([])
            strip_ax.spines["left"].set_visible(False)
            strip_ax.set_ylim(0, 1)

        # hide x tick labels on all but the bottom axis
        for ax in (self._ax, self._ax2, self._ax3, self._ax_gate, self._ax4, self._ax5):
            ax.tick_params(labelbottom=False)

        # time formatter on the bottom-most axis
        _loc = _TimeLocator()
        self._ax_gate2.xaxis.set_major_locator(_loc)
        self._ax_gate2.xaxis.set_major_formatter(mticker.FuncFormatter(_make_fmt(_loc)))

        (self._line,)       = self._ax.plot(      [], [], lw=0.35, color="#111",    rasterized=True)
        (self._line2,)      = self._ax2.plot(     [], [], lw=0.35, color="#4a90d9", rasterized=True)
        (self._line_gate,)  = self._ax_gate.plot( [], [], lw=0.35, color="#111",    rasterized=True)
        (self._line3,)      = self._ax4.plot(     [], [], lw=0.35, color="#27ae60", rasterized=True)
        (self._line_gate2,) = self._ax_gate2.plot([], [], lw=0.35, color="#111",    rasterized=True)

        # threshold lines (hidden until user sets one)
        self._thresh_hline  = self._ax2.axhline(y=0, color="#e74c3c", lw=0.9, ls="--", alpha=0.85, visible=False)
        self._thresh_hline2 = self._ax4.axhline(y=0, color="#e74c3c", lw=0.9, ls="--", alpha=0.85, visible=False)

        # detection event lines
        self._evt_col  = LineCollection([], lw=0.6, color="#e74c3c", rasterized=True)
        self._evt_col2 = LineCollection([], lw=0.6, color="#e74c3c", rasterized=True)
        self._ax3.add_collection(self._evt_col)
        self._ax5.add_collection(self._evt_col2)

        self._uv:            np.ndarray | None = None
        self._neo:           np.ndarray | None = None
        self._neo_avg:       np.ndarray | None = None
        self._avg_w:         int               = 1
        self._threshold:     float | None      = None
        self._threshold_avg: float | None      = None
        self._fs       = 1.0
        self._duration = 0.0
        self._gate_spans:  list = []
        self._gate2_spans: list = []
        self._tc:          int  = 50
        self._wt:          int  = 300
        self._pan_x: float | None = None
        self._pan_lim: tuple | None = None

        # annotation state
        self._annotating   = False
        self._annot_start: float | None = None
        self._annot_preview = None
        self._gt_spans: list = []

        self.mpl_connect("scroll_event",         self._scroll)
        self.mpl_connect("button_press_event",   self._press)
        self.mpl_connect("motion_notify_event",  self._drag)
        self.mpl_connect("button_release_event", self._release)

    # ── annotation control ────────────────────────────────────────────────

    def start_annotating(self):
        self._annotating  = True
        self._annot_start = None

    def cancel_annotating(self):
        self._annotating  = False
        self._annot_start = None
        if self._annot_preview:
            self._annot_preview.remove()
            self._annot_preview = None
        self.draw_idle()

    def set_gt_spans(self, regions):
        for sp in self._gt_spans:
            sp.remove()
        self._gt_spans = []
        for t0, t1 in regions:
            sp = self._ax.axvspan(t0, t1, color="#2ecc71", alpha=0.25, linewidth=0, zorder=2)
            self._gt_spans.append(sp)
        self.draw_idle()

    def set_auto_spans(self, regions):
        self.draw_idle()

    def set_threshold(self, value: float):
        self._threshold = value
        self._thresh_hline.set_ydata([value, value])
        self._thresh_hline.set_visible(True)
        self._rerun_gate()   # calls _redraw() internally

    def set_gate_params(self, tc: int, wt: int):
        self._tc = max(1, tc)
        self._wt = max(1, wt)
        self._rerun_gate()
        self._rerun_gate2()

    def _rerun_gate2(self):
        for sp in self._gate2_spans:
            sp.remove()
        self._gate2_spans = []
        regions = []
        if self._neo_avg is not None and self._threshold_avg is not None:
            valid = ~np.isnan(self._neo_avg)
            triggered = np.where(valid, self._neo_avg > self._threshold_avg, False)
            for s, e in _run_gate(triggered, self._tc, self._wt):
                t0, t1 = s / self._fs, e / self._fs
                sp = self._ax_gate2.axvspan(t0, t1, color="#e74c3c", alpha=0.35, linewidth=0)
                self._gate2_spans.append(sp)
                regions.append((t0, t1))
        self.gate2_updated.emit(regions)
        self._redraw()

    def _rerun_gate(self):
        for sp in self._gate_spans:
            sp.remove()
        self._gate_spans = []
        if self._neo is not None and self._threshold is not None:
            triggered = self._neo > self._threshold
            for s, e in _run_gate(triggered, self._tc, self._wt):
                sp = self._ax_gate.axvspan(
                    s / self._fs, e / self._fs,
                    color="#e74c3c", alpha=0.35, linewidth=0,
                )
                self._gate_spans.append(sp)
        self._redraw()

    def set_avg_threshold(self, value: float):
        self._threshold_avg = value
        self._thresh_hline2.set_ydata([value, value])
        self._thresh_hline2.set_visible(True)
        self._rerun_gate2()   # calls _redraw() internally

    def set_avg_window(self, w: int):
        self._avg_w   = max(1, w)
        if self._neo is not None:
            self._neo_avg = _rolling_mean(self._neo, self._avg_w)
            avg_max = float(np.nanmax(self._neo_avg)) if not np.all(np.isnan(self._neo_avg)) else 1.0
            linthresh = max(avg_max * 0.01, 1.0)
            self._ax4.set_yscale("symlog", linthresh=linthresh)
            self._ax4.set_ylim(0, avg_max * 1.1)
        self._rerun_gate2()   # calls _redraw() internally

    def load(self, uv: np.ndarray, neo: np.ndarray, fs: float):
        self._uv       = uv
        self._neo      = neo
        self._fs       = fs
        self._duration = len(uv) / fs
        pad = (uv.max() - uv.min()) * 0.05 or 1.0
        self._ax.set_ylim(uv.min() - pad, uv.max() + pad)
        for sp in self._gate_spans:
            sp.remove()
        self._gate_spans = []
        self._ax_gate.set_ylim(uv.min() - pad, uv.max() + pad)
        for sp in self._gate2_spans:
            sp.remove()
        self._gate2_spans = []
        self._ax_gate2.set_ylim(uv.min() - pad, uv.max() + pad)
        neo_max = float(neo.max()) or 1.0
        linthresh = max(neo_max * 0.01, 1.0)
        self._ax2.set_yscale("symlog", linthresh=linthresh)
        self._ax2.set_ylim(0, neo_max * 1.1)
        # recompute rolling avg with existing window setting
        if self._avg_w > 1:
            self._neo_avg = _rolling_mean(neo, self._avg_w)
            avg_max = float(np.nanmax(self._neo_avg)) if not np.all(np.isnan(self._neo_avg)) else 1.0
            linthresh2 = max(avg_max * 0.01, 1.0)
            self._ax4.set_yscale("symlog", linthresh=linthresh2)
            self._ax4.set_ylim(0, avg_max * 1.1)
        else:
            self._neo_avg = None
        self._ax.set_xlim(0, self._duration)
        self._redraw()

    def _redraw(self):
        if self._uv is None:
            return
        xmin, xmax = self._ax.get_xlim()
        xmin = max(0.0, xmin)
        xmax = min(self._duration, xmax)

        i0 = int(xmin * self._fs)
        i1 = min(int(xmax * self._fs) + 1, len(self._uv))

        t_c,  uv_lo,  uv_hi  = _envelope(self._uv,  i0, i1, self._fs, MAX_PTS)
        _,    neo_lo, neo_hi = _envelope(self._neo, i0, i1, self._fs, MAX_PTS)

        self._line.set_data(*_envelope_xy(t_c, uv_lo, uv_hi))
        self._line2.set_data(*_envelope_xy(t_c, neo_lo, neo_hi))
        self._line_gate.set_data(*_envelope_xy(t_c, uv_lo, uv_hi))
        self._line_gate2.set_data(*_envelope_xy(t_c, uv_lo, uv_hi))

        def _evt_segs(col, hi_slice, t_centers, threshold):
            if threshold is None:
                col.set_segments([])
                return
            mask = hi_slice > threshold
            et = t_centers[mask]
            n = len(et)
            if n:
                segs = np.empty((n, 2, 2))
                segs[:, 0, 0] = et; segs[:, 0, 1] = 0.05
                segs[:, 1, 0] = et; segs[:, 1, 1] = 0.95
                col.set_segments(segs)
            else:
                col.set_segments([])

        # a bin counts as "triggered" if its max touched the threshold
        # anywhere inside it — catches spikes plain stride-slicing could miss
        _evt_segs(self._evt_col, neo_hi, t_c, self._threshold)

        # rolling-average plot — skip the NaN warm-up region before binning
        if self._neo_avg is not None:
            start = max(i0, self._avg_w - 1)
            if start < i1:
                t_c2, avg_lo, avg_hi = _envelope(self._neo_avg, start, i1, self._fs, MAX_PTS)
                self._line3.set_data(*_envelope_xy(t_c2, avg_lo, avg_hi))
                _evt_segs(self._evt_col2, avg_hi, t_c2, self._threshold_avg)
            else:
                self._line3.set_data([], [])
                self._evt_col2.set_segments([])
        else:
            self._line3.set_data([], [])
            self._evt_col2.set_segments([])

        self._ax.set_xlim(xmin, xmax)
        self.draw_idle()

    def _zoom_around(self, cx, factor):
        xmin, xmax = self._ax.get_xlim()
        span    = max((xmax - xmin) * factor, MIN_SPAN)
        ratio   = (cx - xmin) / (xmax - xmin)
        new_min = max(0.0, cx - span * ratio)
        new_max = new_min + span
        if new_max > self._duration:
            new_max = self._duration
            new_min = max(0.0, new_max - span)
        self._ax.set_xlim(new_min, new_max)
        self._redraw()

    def zoom_key(self, factor):
        if self._uv is None:
            return
        xmin, xmax = self._ax.get_xlim()
        self._zoom_around((xmin + xmax) / 2, factor)

    def pan_step(self, direction):
        """Shift view by one small tick (called from a repeating timer)."""
        if self._uv is None:
            return
        xmin, xmax = self._ax.get_xlim()
        span = xmax - xmin
        step = span * 0.03 * direction          # 3 % of visible span per tick
        new_min = max(0.0, xmin + step)
        new_max = new_min + span
        if new_max > self._duration:
            new_max = self._duration
            new_min = max(0.0, new_max - span)
        self._ax.set_xlim(new_min, new_max)
        self._redraw()

    # ── events ───────────────────────────────────────────────────────────

    def _scroll(self, event):
        if self._uv is None or event.xdata is None:
            return
        self._zoom_around(event.xdata, ZOOM_STEP ** event.step)

    def _press(self, event):
        if event.button != 1 or event.xdata is None:
            return
        if self._annotating:
            self._annot_start = event.xdata
        else:
            self._pan_x   = event.xdata
            self._pan_lim = self._ax.get_xlim()

    def _drag(self, event):
        if event.xdata is None:
            return
        if self._annotating and self._annot_start is not None:
            t0 = min(self._annot_start, event.xdata)
            t1 = max(self._annot_start, event.xdata)
            if self._annot_preview:
                self._annot_preview.remove()
            self._annot_preview = self._ax.axvspan(
                t0, t1, color="#2ecc71", alpha=0.20, linewidth=0, zorder=3
            )
            self.draw_idle()
            return
        if self._pan_x is None:
            return
        lo, hi  = self._pan_lim
        span    = hi - lo
        shift   = self._pan_x - event.xdata
        new_lo  = max(0.0, lo + shift)
        new_hi  = min(self._duration, new_lo + span)
        new_lo  = new_hi - span
        self._ax.set_xlim(new_lo, new_hi)
        self._redraw()

    def _release(self, event):
        if self._annotating and self._annot_start is not None:
            if event.xdata is not None:
                t0 = min(self._annot_start, event.xdata)
                t1 = max(self._annot_start, event.xdata)
                if t1 - t0 > 0.01:
                    self.annotation_done.emit(t0, t1)
            self._annotating  = False
            self._annot_start = None
            if self._annot_preview:
                self._annot_preview.remove()
                self._annot_preview = None
            self.draw_idle()
            return
        self._pan_x = self._pan_lim = None


# ── main window ───────────────────────────────────────────────────────────────

class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("")
        self.setWindowIcon(QIcon(str(_resource("icon.png"))))
        self.resize(1400, 700)
        self._loader: Loader | None = None
        self._cur_path: Path | None = None
        self._pan_dir = 0
        self._pan_timer = QTimer(self)
        self._pan_timer.setInterval(16)
        self._pan_timer.timeout.connect(self._pan_tick)

        self._zoom_dir = 0
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setInterval(16)
        self._zoom_timer.timeout.connect(self._zoom_tick)
        self._annots = _load_annots()   # {str(path): [[t0,t1], ...]}
        self._auto_dets: list = []
        self._build()

    def showEvent(self, event):
        super().showEvent(event)
        if sys.platform == "darwin":
            self._remove_macos_titlebar_separator()

    def _remove_macos_titlebar_separator(self):
        import ctypes, ctypes.util
        try:
            objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            objc.sel_registerName.restype  = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            objc.objc_getClass.restype     = ctypes.c_void_p
            objc.objc_getClass.argtypes    = [ctypes.c_char_p]
            objc.objc_msgSend.restype      = ctypes.c_void_p

            # NSView → NSWindow
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            ns_view   = ctypes.c_void_p(int(self.winId()))
            ns_window = ctypes.c_void_p(
                objc.objc_msgSend(ns_view, objc.sel_registerName(b"window"))
            )

            # Remove separator line
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
            objc.objc_msgSend(ns_window, objc.sel_registerName(b"setTitlebarSeparatorStyle:"),
                              ctypes.c_long(1))   # NSWindowTitlebarSeparatorStyleNone

            # Make title bar area transparent so window background shows through
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            objc.objc_msgSend(ns_window, objc.sel_registerName(b"setTitlebarAppearsTransparent:"),
                              True)

            # Set window background to #252525
            NSColor = ctypes.c_void_p(objc.objc_getClass(b"NSColor"))
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_double, ctypes.c_double,
                                          ctypes.c_double, ctypes.c_double]
            color = ctypes.c_void_p(objc.objc_msgSend(
                NSColor,
                objc.sel_registerName(b"colorWithRed:green:blue:alpha:"),
                ctypes.c_double(0x25 / 255.0),
                ctypes.c_double(0x25 / 255.0),
                ctypes.c_double(0x25 / 255.0),
                ctypes.c_double(1.0),
            ))
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            objc.objc_msgSend(ns_window, objc.sel_registerName(b"setBackgroundColor:"), color)

            # Force dark appearance so chrome stays dark on resize
            NSString = ctypes.c_void_p(objc.objc_getClass(b"NSString"))
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
            dark_name = ctypes.c_void_p(objc.objc_msgSend(
                NSString, objc.sel_registerName(b"stringWithUTF8String:"),
                b"NSAppearanceNameDarkAqua",
            ))
            NSAppearance = ctypes.c_void_p(objc.objc_getClass(b"NSAppearance"))
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            appearance = ctypes.c_void_p(objc.objc_msgSend(
                NSAppearance, objc.sel_registerName(b"appearanceNamed:"), dark_name
            ))
            objc.objc_msgSend(ns_window, objc.sel_registerName(b"setAppearance:"), appearance)

        except Exception:
            pass

    def _start_zoom(self, direction):
        self._zoom_dir = direction
        self._zoom_tick()
        self._zoom_timer.start()

    def _stop_zoom(self):
        self._zoom_timer.stop()
        self._zoom_dir = 0

    def _zoom_tick(self):
        factor = 0.97 if self._zoom_dir > 0 else 1 / 0.97
        self._canvas.zoom_key(factor)
        self._update_zoom_label()

    def _update_zoom_label(self):
        c = self._canvas
        if c._uv is None or c._duration == 0:
            return
        xmin, xmax = c._ax.get_xlim()
        span = max(xmax - xmin, 1e-9)
        zoom = c._duration / span
        text = f"{zoom:.1f}×" if zoom < 10 else f"{zoom:.0f}×"
        self._zoom_label.setText(text)
        self._zoom_label.adjustSize()
        self._reposition_zoom_label()

    def _reposition_zoom_label(self):
        lbl = self._zoom_label
        lbl.move(self._canvas.width() - lbl.width() - 10, 8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._reposition_zoom_label)

    # ── icon-bar style helpers ────────────────────────────────────────────

    _ICON_STYLE = (
        "QPushButton {{ background: {bg}; color: {fg}; border: none;"
        " border-left: 3px solid {bar}; border-radius: 0px; font-size: {sz}px;"
        " padding: 0px; margin: 0px; }}"
        "QPushButton:hover {{ background: rgba(255,255,255,0.10); }}"
    )
    _ICON_SIZES = {"files": 17, "annots": 13, "settings": 22}

    def _icon_style(self, active: bool, key: str = "") -> str:
        sz = self._ICON_SIZES.get(key, 17)
        if active:
            return self._ICON_STYLE.format(
                bg="rgba(255,255,255,0.08)", fg="#fff", bar="#4a90d9", sz=sz)
        return self._ICON_STYLE.format(
            bg="transparent", fg="#888", bar="transparent", sz=sz)

    def _update_icon_styles(self):
        for key, btn in self._icon_btns.items():
            btn.setStyleSheet(self._icon_style(key == self._active_panel, key))
            btn.setFixedSize(42, 42)   # re-assert after every style swap so
                                        # active/inactive states can't drift

    def _toggle_panel(self, key: str):
        if self._active_panel == key:
            self._active_panel = None
            self._side_panel.hide()
        else:
            self._active_panel = key
            self._panel_files.setVisible(key == "files")
            self._panel_annots.setVisible(key == "annots")
            self._panel_settings.setVisible(key == "settings")
            self._side_panel.show()
        self._update_icon_styles()

    # ── build ─────────────────────────────────────────────────────────────

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── main area ─────────────────────────────────────────────────────
        main_area = QWidget()
        root = QHBoxLayout(main_area)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # activity bar (always visible)
        root.addWidget(self._build_icon_bar())

        # collapsible side panel (file tree OR annotation list)
        self._side_panel = QWidget()
        self._side_panel.setFixedWidth(260)
        sp = QVBoxLayout(self._side_panel)
        sp.setContentsMargins(0, 0, 0, 0)
        sp.setSpacing(0)
        self._panel_files    = self._build_file_panel()
        self._panel_annots   = self._build_annot_panel()
        self._panel_settings = self._build_settings_panel()
        self._panel_annots.hide()
        self._panel_settings.hide()
        sp.addWidget(self._panel_files)
        sp.addWidget(self._panel_annots)
        sp.addWidget(self._panel_settings)
        root.addWidget(self._side_panel)

        # main content
        root.addWidget(self._build_main_content(), stretch=1)
        outer.addWidget(main_area, stretch=1)

        # ── footer ────────────────────────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(28)
        footer.setStyleSheet("background: #1e1e1e;")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("© 2026 VerifyHalo")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #555; font-size: 11px; background: transparent;")
        fl.addWidget(lbl)
        outer.addWidget(footer)

        self._active_panel = "files"
        self._update_icon_styles()

        self._canvas.annotation_done.connect(self._on_annotation_done)
        self._canvas.gate2_updated.connect(self._on_gate2_updated)

        if DATA_DIR.exists():
            QTimer.singleShot(0, lambda: self._set_data_dir(DATA_DIR))

    def _build_icon_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedWidth(42)
        bar.setStyleSheet("QWidget { background: #252525; }")
        vbox = QVBoxLayout(bar)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        self._icon_btns: dict = {}
        for key, symbol, tip in (
            ("files",    "≡",  "Files"),
            ("annots",   "◎", "Detections"),
            ("settings", "⊞", "Settings"),
        ):
            btn = QPushButton(symbol)
            btn.setFixedSize(42, 42)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _, k=key: self._toggle_panel(k))
            self._icon_btns[key] = btn
            vbox.addWidget(btn)

        vbox.addStretch(1)
        return bar

    def _build_file_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("QWidget { background-color: #252525; }")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── folder picker row ──────────────────────────────────────────
        picker = QWidget()
        picker.setFixedHeight(40)
        picker.setStyleSheet("QWidget { background: #1e1e1e; border-bottom: 1px solid #333; }")
        pl = QHBoxLayout(picker)
        pl.setContentsMargins(6, 4, 6, 4)
        pl.setSpacing(4)

        self._folder_label = QLabel("No folder selected")
        self._folder_label.setStyleSheet(
            "QLabel { color: #666; font-size: 10px; background: transparent; border: none; }")
        self._folder_label.setMaximumWidth(180)
        self._folder_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        browse_btn = QPushButton("…")
        browse_btn.setFixedSize(24, 24)
        browse_btn.setToolTip("Select folder")
        browse_btn.setStyleSheet(
            "QPushButton { background: #2e2e2e; color: #aaa; border: 1px solid #3a3a3a;"
            " border-radius: 3px; font-size: 13px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.08); color: #fff; }"
            "QPushButton:pressed { background: #1a1a1a; }"
        )
        browse_btn.clicked.connect(self._browse_folder)

        pl.addWidget(self._folder_label, 1)
        pl.addWidget(browse_btn, 0)
        vbox.addWidget(picker)

        # ── file tree ──────────────────────────────────────────────────
        self._fs_model = QFileSystemModel()
        self._fs_model.setNameFilters(["*.rhd"])
        self._fs_model.setNameFilterDisables(False)

        self._tree = QTreeView()
        self._tree.setModel(self._fs_model)
        self._tree.setHeaderHidden(True)
        self._tree.setFrameShape(QFrame.Shape.NoFrame)
        self._tree.setStyleSheet("QTreeView { background: #252525; color: #ccc; border: none; }"
                                  "QTreeView::item:selected { background: #2a4d7a; }"
                                  "QTreeView::item:hover { background: rgba(255,255,255,0.05); }")
        for col in (1, 2, 3):
            self._tree.hideColumn(col)
        self._tree.clicked.connect(self._on_tree_click)
        vbox.addWidget(self._tree, 1)

        return container

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select data folder",
                                                 str(Path.home()))
        if path:
            self._set_data_dir(Path(path))

    def _set_data_dir(self, folder: Path):
        self._fs_model.setRootPath(str(folder))
        self._tree.setRootIndex(self._fs_model.index(str(folder)))
        name = folder.name or str(folder)
        self._folder_label.setText(name)
        self._folder_label.setToolTip(str(folder))
        self._folder_label.setStyleSheet(
            "QLabel { color: #ccc; font-size: 10px; background: transparent; }")
        # auto-open first rhd in the new folder
        first = next(iter(sorted(folder.rglob("*.rhd"))), None)
        if first:
            self._open(first)

    def _build_main_content(self) -> QWidget:
        right = QWidget()
        right.setStyleSheet("background: #ffffff;")
        vbox  = QVBoxLayout(right)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        hdr_widget = QWidget()
        hdr_widget.setStyleSheet("""
            QWidget   { background: #252525; }
            QLabel    { color: #888; font-size: 12px; background: transparent; }
            QComboBox { background: #2e2e2e; color: #ccc; border: 1px solid #3a3a3a;
                        border-radius: 3px; padding: 2px 4px; font-size: 12px; }
            QComboBox::drop-down { border: none; width: 14px; }
            QComboBox QAbstractItemView { background: #2e2e2e; color: #ccc;
                        selection-background-color: #3a3a3a;
                        min-width: 80px; max-width: 80px; }
            QLineEdit { background: #2e2e2e; color: #ccc; border: 1px solid #3a3a3a;
                        border-radius: 3px; padding: 2px 6px; font-size: 12px; }
            QPushButton { background: #2e2e2e; color: #aaa; border: 1px solid #3a3a3a;
                          border-radius: 3px; font-size: 12px; }
            QPushButton:hover   { background: rgba(255,255,255,0.08); color: #fff; }
            QPushButton:pressed { background: #1a1a1a; }
            QPushButton#apply_btn { background: #4a90d9; color: #fff;
                          border: 1px solid #3a7fbf; border-radius: 3px; font-size: 12px; }
            QPushButton#apply_btn:hover   { background: #5aa0e9; }
            QPushButton#apply_btn:pressed { background: #3a7fbf; }
        """)
        hdr_widget.setFixedHeight(41)
        hdr = QHBoxLayout(hdr_widget)
        hdr.setContentsMargins(8, 0, 8, 0)
        hdr.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        _H = 26
        _block_style = (
            "QLabel { background: #2e2e2e; color: #ccc; border: 1px solid #3a3a3a;"
            " border-radius: 3px; padding: 2px 8px; font-size: 12px; }"
        )

        info_grp = QWidget()
        info_grp.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        ig = QHBoxLayout(info_grp)
        ig.setContentsMargins(0, 0, 0, 0)
        ig.setSpacing(6)
        self._ch_combo = QComboBox()
        self._ch_combo.setEnabled(False)
        self._ch_combo.setFixedWidth(80)
        self._ch_combo.currentIndexChanged.connect(self._on_channel_change)
        ig.addWidget(self._ch_combo, 0, Qt.AlignmentFlag.AlignVCenter)
        self._day_label = QLabel("Day —")
        self._day_label.setFixedHeight(_H)
        self._day_label.setStyleSheet(_block_style)
        ig.addWidget(self._day_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self._hour_label = QLabel("Hour —")
        self._hour_label.setFixedHeight(_H)
        self._hour_label.setStyleSheet(_block_style)
        ig.addWidget(self._hour_label, 0, Qt.AlignmentFlag.AlignVCenter)
        info_grp.adjustSize()
        hdr.addWidget(info_grp, 0, Qt.AlignmentFlag.AlignVCenter)
        hdr.addStretch()
        vbox.addWidget(hdr_widget, stretch=0)

        self._canvas = WaveCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._canvas.setFixedHeight(940)

        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: #ffffff; border: none; }
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 0, 0, 0.28);
                border-radius: 3px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 0, 0, 0.45);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
                height: 0px;
            }
        """)
        vbox.addWidget(scroll, stretch=1)

        btn_style = (
            "QPushButton { background: #2e2e2e; color: #aaa; border: 1px solid #3a3a3a;"
            " border-radius: 3px; font-size: 15px; font-weight: bold; }"
            "QPushButton:pressed { background: #1a1a1a; }"
        )
        # overlay: two rows of buttons, explicit fixed size so nothing overlaps
        # row heights: 6(top pad) + 26(btn) + 6(gap) + 26(btn) + 4(bottom pad) = 68
        BTN = 26
        overlay = QWidget(self._canvas)
        overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        overlay.setFixedSize(80, 65)

        ov_layout = QVBoxLayout(overlay)
        ov_layout.setContentsMargins(8, 8, 0, 4)
        ov_layout.setSpacing(3)

        ol = QHBoxLayout()
        ol.setSpacing(3)
        for symbol, direction in (("+", +1), ("−", -1)):
            btn = QPushButton(symbol)
            btn.setFixedSize(BTN, BTN)
            btn.setStyleSheet(btn_style)
            btn.pressed.connect(lambda d=direction: self._start_zoom(d))
            btn.released.connect(self._stop_zoom)
            ol.addWidget(btn)
        ol.addStretch()
        ov_layout.addLayout(ol)

        ol2 = QHBoxLayout()
        ol2.setSpacing(3)
        for symbol, direction in (("‹", -1), ("›", +1)):
            btn = QPushButton(symbol)
            btn.setFixedSize(BTN, BTN)
            btn.setStyleSheet(btn_style)
            btn.pressed.connect(lambda d=direction: self._start_pan(d))
            btn.released.connect(self._stop_pan)
            ol2.addWidget(btn)
        ol2.addStretch()
        ov_layout.addLayout(ol2)

        overlay.move(0, 0)

        self._zoom_label = QLabel("1×", self._canvas)
        self._zoom_label.setStyleSheet(
            "color: #888; font-size: 11px; background: transparent;")
        self._zoom_label.adjustSize()

        return right

    # ── interactions ──────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if event.isAutoRepeat():
            return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._start_zoom(+1)
        elif key == Qt.Key.Key_Minus:
            self._start_zoom(-1)
        elif key == Qt.Key.Key_Right:
            self._start_pan(+1)
        elif key == Qt.Key.Key_Left:
            self._start_pan(-1)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()
        if event.isAutoRepeat():
            return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus):
            self._stop_zoom()
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_Left):
            self._stop_pan()
        else:
            super().keyReleaseEvent(event)

    def _start_pan(self, direction):
        self._pan_dir = direction
        self._pan_tick()            # immediate first step — no lag on press
        self._pan_timer.start()

    def _stop_pan(self):
        self._pan_timer.stop()
        self._pan_dir = 0

    def _pan_tick(self):
        self._canvas.pan_step(self._pan_dir)

    def _on_tree_click(self, index: QModelIndex):
        path = Path(self._fs_model.filePath(index))
        if path.is_file() and path.suffix == ".rhd":
            self._open(path)

    def _open(self, path: Path):
        prev_ch = self._ch_combo.currentIndex()   # -1 if nothing was selected yet
        self._cur_path = path
        day_m  = re.search(r'\d+', path.parent.name)
        # filenames look like "BCI_260504_145436_1 hr.rhd" — the hour number
        # is the last "_<digits> hr" segment before the extension, with a
        # space before "hr" (not glued to the digits like "1hr")
        hour_m = re.search(r'_(\d+)\s*hr$', path.stem)
        day  = day_m.group()  if day_m  else path.parent.name
        hour = hour_m.group(1) if hour_m else path.stem
        self._day_label.setText(f"Day #{day}")
        self._day_label.adjustSize()
        self._hour_label.setText(f"Hour #{hour}")
        self._hour_label.adjustSize()
        QApplication.processEvents()
        try:
            n_ch, fs = read_rhd_info(path)
        except Exception as e:
            return

        self._ch_combo.blockSignals(True)
        self._ch_combo.clear()
        for i in range(n_ch):
            self._ch_combo.addItem(f"CH #{i}")
            self._ch_combo.setItemData(
                i, Qt.AlignmentFlag.AlignCenter, Qt.ItemDataRole.TextAlignmentRole)
        self._ch_combo.setEnabled(True)
        # keep whatever channel was selected on the previous file, clamped
        # to this file's channel count (falls back to 0 on first open or if
        # the new file has fewer channels)
        new_ch = prev_ch if 0 <= prev_ch < n_ch else 0
        self._ch_combo.setCurrentIndex(new_ch)
        self._ch_combo.blockSignals(False)

        self._load_channel(new_ch)

    def _on_channel_change(self, idx: int):
        if self._cur_path and idx >= 0:
            self._load_channel(idx)

    def _load_channel(self, ch: int):
        if not self._cur_path:
            return
        if self._loader and self._loader.isRunning():
            self._loader.terminate()
            self._loader.wait()


        self._loader = Loader(self._cur_path, ch)
        self._loader.done.connect(self._on_loaded)
        self._loader.error.connect(self._on_error)
        self._loader.start()

    def _on_loaded(self, uv, neo, fs, ch, auto_dets):
        self._canvas.load(uv, neo, fs)
        self._auto_dets = []
        gt = self._annots.get(str(self._cur_path), [])
        self._canvas.set_gt_spans(gt)
        self._refresh_annot_panel()
        self._zoom_label.setText("1×")
        self._zoom_label.adjustSize()
        self._reposition_zoom_label()
        self._on_threshold_apply()

    def _on_threshold_apply(self):
        try:
            self._canvas.set_threshold(float(self._thresh_edit.text()))
        except ValueError:
            pass
        try:
            w = int(self._avg_edit.text())
            if w > 0:
                self._canvas.set_avg_window(w)
        except ValueError:
            pass
        try:
            self._canvas.set_avg_threshold(float(self._avg_thresh_edit.text()))
        except ValueError:
            pass
        _tc = self._canvas._tc
        _wt = self._canvas._wt
        gate_changed = False
        try:
            v = int(self._tc_edit.text())
            if v > 0:
                _tc = v
                gate_changed = True
        except ValueError:
            pass
        try:
            v = int(self._wt_edit.text())
            if v > 0:
                _wt = v
                gate_changed = True
        except ValueError:
            pass
        if gate_changed:
            self._canvas.set_gate_params(_tc, _wt)

    # ── annotation panel ─────────────────────────────────────────────────

    def _build_annot_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(320)
        panel.setStyleSheet("QWidget { background-color: #252525; }")
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        _btn_style = (
            "QPushButton { background: #2e2e2e; color: #aaa; border: 1px solid #3a3a3a;"
            " border-radius: 3px; font-size: 12px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.08); color: #fff; }"
            "QPushButton:pressed { background: #1a1a1a; }"
            "QPushButton:disabled { background: #252525; color: #555; border-color: #2e2e2e; }"
        )
        # Annotate / Cancel buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._btn_annotate = QPushButton("Annotate")
        self._btn_cancel   = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        for b in (self._btn_annotate, self._btn_cancel):
            b.setFixedHeight(26)
            b.setStyleSheet(_btn_style)
            btn_row.addWidget(b)
        self._btn_annotate.clicked.connect(self._on_annotate)
        self._btn_cancel.clicked.connect(self._on_cancel_annotate)
        vbox.addLayout(btn_row)

        # Scroll area with two sections
        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._annot_container = QWidget()
        self._annot_layout = QVBoxLayout(self._annot_container)
        self._annot_layout.setContentsMargins(4, 0, 4, 0)
        self._annot_layout.setSpacing(4)
        self._annot_layout.addStretch(1)
        scroll.setWidget(self._annot_container)
        vbox.addWidget(scroll)
        return panel

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(320)
        panel.setStyleSheet("QWidget { background-color: #252525; }")
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(4)

        _lbl_style  = "QLabel { font-size: 10px; color: #888; background: transparent; }"
        _edit_style = (
            "QLineEdit { background: #2e2e2e; color: #ccc; border: 1px solid #3a3a3a;"
            " border-radius: 3px; padding: 2px 6px; font-size: 12px; }"
        )
        _reload_style = (
            "QPushButton { background: #4a90d9; color: #fff; border: 1px solid #3a7fbf;"
            " border-radius: 3px; font-size: 12px; }"
            "QPushButton:hover   { background: #5aa0e9; }"
            "QPushButton:pressed { background: #3a7fbf; }"
        )

        def _field(attr, label, default):
            lbl = QLabel(label)
            lbl.setStyleSheet(_lbl_style)
            edit = QLineEdit()
            edit.setText(default)
            edit.setStyleSheet(_edit_style)
            edit.setFixedHeight(26)
            edit.returnPressed.connect(self._on_threshold_apply)
            vbox.addWidget(lbl)
            vbox.addWidget(edit)
            vbox.addSpacing(4)
            setattr(self, attr, edit)

        _field("_thresh_edit",     "NEO threshold (µV²)",  "150000")
        _field("_avg_edit",        "avg window (samples)",  "30000")
        _field("_avg_thresh_edit", "AVG threshold (µV²)",  "20000")
        _field("_tc_edit",         "transition count",      "10")
        _field("_wt_edit",         "window timeout",        "1000")

        vbox.addSpacing(4)
        self._thresh_btn = QPushButton("Reload")
        self._thresh_btn.setFixedHeight(28)
        self._thresh_btn.setStyleSheet(_reload_style)
        self._thresh_btn.clicked.connect(self._on_threshold_apply)
        vbox.addWidget(self._thresh_btn)
        vbox.addStretch(1)
        return panel

    def _refresh_annot_panel(self):
        # clear existing rows (keep the trailing stretch)
        while self._annot_layout.count() > 1:
            item = self._annot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        def _section(title):
            lbl = QLabel(title)
            lbl.setStyleSheet(
                "QLabel { font-size: 10px; color: #888; background: transparent;"
                " padding: 6px 2px 2px 2px; }")
            self._annot_layout.insertWidget(self._annot_layout.count() - 1, lbl)

        def _row(t0, t1, color, on_delete=None):
            row = QWidget()
            row.setStyleSheet(
                "QWidget { background: #2e2e2e; border-radius: 3px; }")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(6, 4, 4, 4)
            rl.setSpacing(6)
            dot = QLabel("●")
            dot.setStyleSheet(
                f"QLabel {{ color: {color}; font-size: 9px; background: transparent; }}")
            rl.addWidget(dot)
            dur = _fmt_abs(max(0, int(round(t1 - t0))))
            txt = QLabel(f"{_fmt_abs(int(t0))} → {_fmt_abs(int(t1))}  ({dur})")
            txt.setStyleSheet(
                "QLabel { font-size: 11px; color: #ccc; background: transparent; }")
            rl.addWidget(txt, 1)
            if on_delete:
                btn = QPushButton("×")
                btn.setFixedSize(16, 16)
                btn.setStyleSheet(
                    "QPushButton { background: transparent; color: #666;"
                    " border: none; font-size: 13px; font-weight: bold; }"
                    "QPushButton:hover { color: #e74c3c; }"
                )
                btn.clicked.connect(on_delete)
                rl.addWidget(btn)
            self._annot_layout.insertWidget(self._annot_layout.count() - 1, row)

        # Ground truth
        _section("Ground Truth")
        gt = self._annots.get(str(self._cur_path), []) if self._cur_path else []
        for i, (t0, t1) in enumerate(gt):
            idx = i
            _row(t0, t1, "#2ecc71",
                 on_delete=lambda _, i=idx: self._on_delete_gt(i))

        # Auto-detected
        _section("Auto-detected")
        for t0, t1 in self._auto_dets:
            _row(t0, t1, "#e67e22")

    def _on_gate2_updated(self, regions):
        self._auto_dets = regions
        self._refresh_annot_panel()

    def _on_annotate(self):
        self._canvas.start_annotating()
        self._btn_annotate.setEnabled(False)
        self._btn_cancel.setEnabled(True)

    def _on_cancel_annotate(self):
        self._canvas.cancel_annotating()
        self._btn_annotate.setEnabled(True)
        self._btn_cancel.setEnabled(False)

    def _on_annotation_done(self, t0: float, t1: float):
        self._btn_annotate.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        if not self._cur_path:
            return
        key = str(self._cur_path)
        self._annots.setdefault(key, []).append([t0, t1])
        _save_annots(self._annots)
        gt = self._annots[key]
        self._canvas.set_gt_spans(gt)
        self._refresh_annot_panel()

    def _on_delete_gt(self, idx: int):
        if not self._cur_path:
            return
        key = str(self._cur_path)
        regions = self._annots.get(key, [])
        if 0 <= idx < len(regions):
            regions.pop(idx)
            self._annots[key] = regions
            _save_annots(self._annots)
            self._canvas.set_gt_spans(regions)
            self._refresh_annot_panel()

    def _on_error(self, msg):
        pass


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("RHD Viewer")
    app.setWindowIcon(QIcon(str(_resource("icon.png"))))

    if sys.platform == "darwin":
        try:
            import ctypes, ctypes.util
            _objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            _objc.sel_registerName.restype  = ctypes.c_void_p
            _objc.sel_registerName.argtypes = [ctypes.c_char_p]
            _objc.objc_getClass.restype     = ctypes.c_void_p
            _objc.objc_getClass.argtypes    = [ctypes.c_char_p]
            _objc.objc_msgSend.restype      = ctypes.c_void_p

            _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
            _ns_name = ctypes.c_void_p(_objc.objc_msgSend(
                ctypes.c_void_p(_objc.objc_getClass(b"NSString")),
                _objc.sel_registerName(b"stringWithUTF8String:"),
                b"RHD Viewer",
            ))
            _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            _proc_info = ctypes.c_void_p(_objc.objc_msgSend(
                ctypes.c_void_p(_objc.objc_getClass(b"NSProcessInfo")),
                _objc.sel_registerName(b"processInfo"),
            ))
            _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            _objc.objc_msgSend(_proc_info, _objc.sel_registerName(b"setProcessName:"), _ns_name)
        except Exception:
            pass
    win = Viewer()
    win.show()
    sys.exit(app.exec())
