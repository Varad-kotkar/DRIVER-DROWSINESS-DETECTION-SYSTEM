# Driver Drowsiness Detection System

Real-time driver drowsiness detection from a webcam feed, using facial
landmark tracking and the **Eye Aspect Ratio (EAR)** technique.

## How it works

1. **Face landmark detection** — [MediaPipe FaceMesh](https://developers.google.com/mediapipe)
   locates 468 facial landmarks per frame, including precise points around
   each eye.
2. **Eye Aspect Ratio (EAR)** — for each eye, EAR compares the vertical
   distance between the eyelids to the horizontal width of the eye:

   ```
   EAR = (‖p2 - p6‖ + ‖p3 - p5‖) / (2 * ‖p1 - p4‖)
   ```

   EAR stays roughly constant while eyes are open and drops sharply when
   they close. This is the standard method introduced by Soukupova & Cech
   (2016) and is more reliable than plain Haar-cascade eye detection, since
   it works continuously across head angle, glasses, and lighting changes
   rather than doing a binary "eye found / not found" check per frame.
3. **Time-based alert, not frame-based** — the system tracks *how long*
   EAR has stayed below the threshold using a wall-clock timer, not a raw
   frame count. A frame-count threshold (e.g. "20 frames") is unreliable
   because it silently changes meaning with camera FPS — 20 frames is under
   a second on a 30fps camera but several seconds on a slow one. Here,
   `--closed-seconds` means the same thing regardless of hardware.
4. **Face-missing vs. eyes-closed are tracked separately** — no face
   detected (e.g. driver turned away) triggers a different warning than
   eyes verifiably closed, instead of treating both as the same signal.

## Demo

<img width="1440" height="1440" alt="compile" src="https://github.com/user-attachments/assets/d0a777c3-a8bd-4178-8856-34d896a7572d" />
![Uploading compile.jpg…]()
<img width="1440" height="1440" alt="compile" src="https://github.com/user-attachments/assets/befcdfb4-2542-4169-9a0e-745b05afee49" />
![Uploading compile.jpg…]()



## Setup

```bash
git clone https://github.com/Varad-kotkar/DRIVER-DROWSINESS-DETECTION-SYSTEM.git
cd driver-drowsiness-detection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
cd src
python drowsiness_detector.py
```

Optional flags:

```bash
python drowsiness_detector.py --camera 1 --ear-threshold 0.22 --closed-seconds 1.2
```

| Flag | Default | Meaning |
|---|---|---|
| `--camera` | `0` | Webcam index to use |
| `--ear-threshold` | `0.25` | EAR value below which eyes are considered closed |
| `--closed-seconds` | `1.5` | How many seconds eyes must stay closed before alerting |

Press `q` to quit.

## Project structure

```
driver-drowsiness-detection/
├── src/
│   ├── drowsiness_detector.py   # main application loop, camera, alerts
│   └── ear_utils.py             # EAR math, isolated and unit-testable
├── requirements.txt
└── README.md
```

## Known limitations & next steps

- Single-face only (`max_num_faces=1`) — fine for a driver-facing camera,
  not designed for multiple people in frame.
- Sound alert is best-effort: uses `winsound` on Windows and a terminal
  bell elsewhere; the on-screen red banner is the primary, guaranteed
  alert on every platform.
- No head-pose estimation yet — a driver nodding off with their head
  tilted down can partially fool landmark detection. Adding head-pose
  angle as a second signal alongside EAR would improve robustness.
- No logging/analytics of drowsiness events over a session — could add a
  simple CSV/SQLite log of alert timestamps for a "session report" feature.
- Threshold values (`0.25`, `1.5s`) are reasonable defaults from published
  EAR research, not tuned against a labeled drowsiness dataset — a good
  future step would be calibrating thresholds against a real dataset
  (e.g. NTHU-DDD) and reporting precision/recall.
