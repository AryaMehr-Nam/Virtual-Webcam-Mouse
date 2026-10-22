# ✋ Hand Gesture Virtual Mouse v3.0

> Control your PC with hand gestures — no physical mouse needed!

Built with **MediaPipe Tasks API** + **OpenCV** + **PyAutoGUI**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.30+-green)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)

---

## 🎯 Gestures

| Gesture | How to do it | Action |
|---------|-------------|--------|
| ☝️ **Index up** | Raise only index finger | **Move cursor** |
| 👌 **Thumb + Index pinch** | Pinch thumb & index tips | **Left click** (cursor keeps moving!) |
| 🤏 **Thumb + Middle pinch** | Pinch thumb & middle tips | **Right click** |
| 👍 **Thumb only** | Only thumb extended | **Double click** |
| ✌️ **Index + Middle** | Peace sign | **Scroll** (move hand up/down) |
| ✊ **Fist** | Close all fingers | **Drag & drop** |
| 🖐️ **Open palm** | All 5 fingers up | **Pause** (safe zone, no action) |

## ⚡ Features

- **Click while moving** — cursor doesn't freeze during pinch
- **Robust finger detection** — uses curl geometry, not simple Y-comparison
- **Smart thumb detection** — uses X-axis with handedness awareness
- **Reliable fist detection** — multi-check curl + MCP distance validation
- **Normalized pinch distances** — works at any distance from camera
- **Adaptive smoothing** — responsive yet stable cursor
- **Debug overlay** — press `D` to see all detection values live
- **Auto model download** — downloads MediaPipe model on first run

## 📦 Installation

```bash
git clone https://github.com/AryaMehr-Nam/Virtual-Webcam-Mouse.git
cd Virtual-Webcam-Mouse
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- Webcam
- Windows OS

## 🚀 Usage

```bash
python hand_mouse.py
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `P` | Pause / Resume |
| `D` | Toggle debug overlay |

## ⚙️ Tuning

Edit the config section at the top of `hand_mouse.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `PINCH_THRESH` | `0.06` | Pinch sensitivity (lower = need closer pinch) |
| `SMOOTHING` | `0.25` | Cursor smoothing (lower = smoother but laggy) |
| `SCROLL_SENS` | `3.0` | Scroll speed multiplier |
| `MARGIN` | `100` | Active zone margin from edges |
| `CLICK_CD` | `0.4` | Cooldown between clicks (seconds) |

## 🏗️ Architecture

```
hand_mouse.py          # Main application
download_model.py      # Resumable model downloader (for slow connections)
hand_landmarker.task   # MediaPipe model (auto-downloaded)
requirements.txt       # Python dependencies
README.md              # This file
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model download fails | Run `python download_model.py` (supports resume) |
| Cursor jittery | Decrease `SMOOTHING` value (e.g. `0.15`) |
| Clicks too sensitive | Increase `PINCH_THRESH` (e.g. `0.08`) |
| Wrong hand detected | Code auto-detects handedness, try other hand |
| Low FPS | Reduce `CAM_W`/`CAM_H` to `640x480` |

## 📄 License

MIT License — free to use, modify, and distribute.
