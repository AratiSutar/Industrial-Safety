# Industrial Safety Monitor

[![CI](https://github.com/AratiSutar/Industrial-Safety/actions/workflows/ci.yml/badge.svg)](https://github.com/AratiSutar/Industrial-Safety/actions/workflows/ci.yml)

Camera-based safety assistant for a factory:

- **Gate mode:** checks PPE (hard hat, high-vis vest, safety glasses, safety boots) before a worker enters.
- **Factory mode:** measures each person's floor distance to every running machine and raises
  alerts in levels: **WARN** (amber light at the machine) → **ALERT** (operator screen; operator
  announces by location, e.g. "Person at Press 3, step back") → **ALARM** (buzzer at that machine only).

> **Not a certified safety device.** This system adds warnings on top of guards, light curtains
> and emergency stops (ISO 13849 / IEC 62061). It must never replace them or stop machines
> without review by a safety engineer.

## Privacy by design

- Video frames stay in memory and are never written to disk by default.
- Workers are not identified: no face recognition, no names, no lasting IDs. Alerts refer to a location.
- Faces are blurred in anything shown or stored. Snapshots (if enabled) are deleted automatically.
- A DPIA and sign-off from Legal and worker representatives are required before filming real staff.

## License note

Ultralytics YOLO is licensed under AGPL-3.0. Internal use and prototypes are fine; check whether
an Ultralytics commercial license is needed before distributing or offering this as a service.

## Setup (Windows, CPU only)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env          # then add your TypeSafe API key
copy config\machines.example.yaml config\machines.yaml
```

## Run

```powershell
python main.py --source 0 --show           # laptop webcam with preview window (faces blurred)
python main.py --source path\to\video.mp4  # video file
python main.py --backend pytorch           # compare speed against OpenVINO
python main.py --mode GATE --fake-jev      # test the background Jev pipeline offline
python tools/calibrate.py --camera cam_floor_1 --source 0   # calibrate a camera (see below)
python -m pytest                           # tests
ruff check .                               # lint
```

## CI

Every push to `main` and every pull request runs [.github/workflows/ci.yml](.github/workflows/ci.yml):
lint (ruff), unit tests (pytest), and a smoke test of the tracker on a synthetic video.
CI never uses real factory footage. A Docker image build and publish step (CD) is added with the Docker milestone.

On first run the pose model (`yolov8n-pose.pt`) is downloaded into `models/` and converted to
OpenVINO format once. In the preview, a green dot marks where a person stands (from the ankles);
an orange dot means the ankles were hidden and the bottom of the person's box was used instead.

## Camera calibration

Floor distances need one calibration per camera:

1. Mark at least 4 points on the floor with tape (5–6 is better), spread out, never 3 on one line.
2. Measure each point from one corner of the room: x to the right, y forward, in meters.
3. Run `python tools/calibrate.py --camera <camera id> --source <camera>`, click each point and type its position.
4. Check the reported error (aim for under 10 cm) and that the blue 1 m grid follows the floor, then press `s`.

The result is saved to `calibration/<camera>.json` (not committed: it is site-specific).
Machine areas in `config/machines.yaml` use the same floor coordinates.

## Jev (PPE checks)

Each PPE item gets its own yes / no / unclear question to Jev, with its own crop, sent in parallel
from background threads so the video loop never waits for the network. The gate only unlocks
when every required item is a confident "yes"; "no", "unclear", errors and timeouts keep it locked.
The hard-hat crop stops above the eyes and the other crops come from the face-blurred frame.
Safety glasses are not sent to Jev; they will be checked locally.

## Project layout

```
main.py                    orchestrator
safety/config.py           loads settings.yaml, machines.yaml and .env
safety/video.py            webcam / file / RTSP input with frame skipping
safety/detector.py         Block 1: YOLOv8n-pose tracking, foot point from ankles
safety/privacy.py          face blurring (blurs the whole head zone when unsure)
safety/display.py          drawing for the local preview
safety/homography.py       Block 2: pixels -> floor meters, per-camera calibration files
safety/zones.py            distance from a person to the nearest machine area
safety/crops.py            per-item PPE crops sized for Jev (280-1568 px), no faces
safety/jev_client.py       Block 3: TypeSafe Jev client (+ offline fake)
safety/inspector.py        background Jev checks and the fail-closed gate decision
safety/gate.py             detects when a person is standing still in the gate box
tools/calibrate.py         interactive camera calibration
config/settings.yaml       general settings (committed)
config/machines.yaml       your factory layout (not committed; copy from the example)
models/                    model weights (not committed)
```

## Roadmap

1. ✅ Project skeleton
2. ✅ Pose tracking and ankle point (YOLOv8n-pose)
3. ✅ Camera calibration, floor distance, live-camera fix, background Jev inspector
4. Proximity alert levels (WARN / ALERT / ALARM) and machine running state
5. Operator screen and event log
6. Gate: local safety-glasses check, test with real Jev API key
7. Docker setup and CD (image published to GitHub Container Registry)
