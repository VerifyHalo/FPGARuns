# RHD Viewer

A desktop viewer and seizure annotation interface for Intan RHD2000 neural recording files, built for the VerifyHalo FPGA seizure detection pipeline.

![RHD Viewer demo](demo.png)

---

## Download & Install

Go to the [latest release](https://github.com/VerifyHalo/FPGARuns/releases/latest) and download the file for your platform.

### Windows

1. Download `RHD Viewer.exe`
2. Double-click to launch — no installation required

> Windows may show a SmartScreen warning on first launch because the app is not yet code-signed. Click **More info → Run anyway** to proceed.

### macOS

1. Download `RHD Viewer.dmg`
2. Open the `.dmg` file
3. Drag **RHD Viewer** into your Applications folder
4. Eject the disk image

**First launch (Gatekeeper):** Because the app is not yet notarized with an Apple Developer certificate, macOS will block it on the first open. To get past this (one time only):

1. Right-click (or Control-click) the app in Applications
2. Choose **Open**
3. Click **Open** in the dialog that appears

After this the app opens normally every time.

---

## Usage

1. Launch RHD Viewer
2. Click the **≡** icon in the left bar and use the **…** button to select the folder containing your `.rhd` files
3. Click any file in the tree to load it
4. Open the Settings panel (**⊞**) to adjust detection parameters and click **Reload**
