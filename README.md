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

## Project layout

```
main.py                    orchestrator
safety/config.py           loads settings.yaml, machines.yaml and .env
safety/video.py            webcam / file / RTSP input with frame skipping
safety/detector.py         Block 1: YOLOv8n-pose tracking, foot point from ankles
safety/privacy.py          face blurring (blurs the whole head zone when unsure)
safety/display.py          drawing for the local preview
config/settings.yaml       general settings (committed)
config/machines.yaml       your factory layout (not committed; copy from the example)
models/                    model weights (not committed)
```

## Roadmap

1. ✅ Project skeleton
2. ✅ Pose tracking and ankle point (YOLOv8n-pose)
3. Camera calibration and floor distance (homography)
4. Proximity alert levels and machine config
5. Operator screen and event log
6. Jev client and gate logic
7. Docker setup
