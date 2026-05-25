# RHD Viewer

A desktop viewer for Intan RHD2000 neural recording files, built for the VerifyHalo FPGA seizure detection pipeline.

> **Platform:** macOS only (Apple Silicon & Intel)

---

## Download & Install

1. Go to the [**latest release**](https://github.com/VerifyHalo/FPGARuns/releases/latest)
2. Download **RHD Viewer.dmg**
3. Open the `.dmg` file
4. Drag **RHD Viewer** into your **Applications** folder
5. Eject the disk image

### First launch (Gatekeeper)

Because the app is not yet notarized with an Apple Developer certificate, macOS will block it on the first open. To get past this — **one time only**:

1. **Right-click** (or Control-click) the app in Applications
2. Choose **Open**
3. Click **Open** in the dialog that appears

After this the app opens normally every time.

---

## Usage

1. Launch **RHD Viewer**
2. Click the **folder icon (≡)** in the left bar and use the **…** button to select the folder containing your `.rhd` files
3. Click any file in the tree to load it
4. Open the **Settings panel (⚙)** to adjust detection parameters and click **Reload**

### Plots (top → bottom)

| Plot | Description |
|------|-------------|
| **µV** | Raw signal — annotations drawn here (green) |
| **\|NEO\| µV²** | Non-linear energy operator |
| **det.** | Per-sample NEO threshold crossings |
| **µV (gate)** | Raw signal + NEO gate detections (red) |
| **\|NEO avg\| µV²** | Rolling-average smoothed NEO |
| **det.** | Per-sample AVG threshold crossings |
| **µV (avg gate)** | Raw signal + AVG gate detections (red) |

### Detection parameters

| Field | Description |
|-------|-------------|
| NEO threshold (µV²) | Threshold applied to the raw NEO signal |
| avg window (samples) | Rolling-average window length |
| AVG threshold (µV²) | Threshold applied to the smoothed NEO |
| transition count | Consecutive detections required to enter seizure state |
| window timeout | Consecutive non-detections required to exit seizure state |

### Navigation

| Action | Control |
|--------|---------|
| Zoom in / out | Scroll wheel · `+` / `-` keys · on-screen buttons |
| Pan | Click-drag · `←` / `→` keys · on-screen arrows |
| Annotate | Open **Detections (◎)** panel → **Annotate** → click-drag on plot |

---

## Running from source

Requires Python 3.11+ and the packages below.

```bash
pip install PyQt6 matplotlib numpy
python viewer.py
```

---

## License

© 2026 VerifyHalo
