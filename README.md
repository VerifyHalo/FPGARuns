# RHD Viewer

A desktop viewer and seizure anotation interface for Intan RHD2000 neural recording files, built for the VerifyHalo FPGA seizure detection pipeline.

> Platform: macOS only (Apple Silicon & Intel)

![RHD Viewer demo](demo.png)

---

## Download & Install

1. Go to the [latest release](https://github.com/VerifyHalo/FPGARuns/releases/latest)
2. Download RHD Viewer.dmg
3. Open the `.dmg` file
4. Drag RHD Viewer into your Applications folder
5. Eject the disk image

### First launch (Gatekeeper)

Because the app is not yet notarized with an Apple Developer certificate, macOS will block it on the first open. To get past this (one time only):

1. Right-click (or Control-click) the app in Applications
2. Choose Open
3. Click Open in the dialog that appears

After this the app opens normally every time.

---

## Usage

1. Launch **RHD Viewer**
2. Click the **Folder icon** in the left bar and use the **…** button to select the folder containing your `.rhd` files
3. Click any file in the tree to load it
4. Open the **Settings panel** to adjust detection parameters and click **Reload**
