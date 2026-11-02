# Industrial Safety Monitor

[![CI](https://github.com/AratiSutar/Industrial-Safety/actions/workflows/ci.yml/badge.svg)](https://github.com/AratiSutar/Industrial-Safety/actions/workflows/ci.yml)

Camera-based safety assistant for a factory:

- **Gate mode:** checks PPE (hard hat, high-vis vest, safety boots) before a worker enters.
  A safety-glasses check is planned (see Roadmap).
- **Factory mode:** measures each person's floor distance to every running machine and raises
  alerts in levels: **WARN** (amber light at the machine) → **ALERT** (operator screen; operator
  announces by location, e.g. "Person at Press 3, step back") → **ALARM** (buzzer at that machine only).

> **Not a certified safety device.** This system adds warnings on top of guards, light curtains
> and emergency stops (ISO 13849 / IEC 62061). It must never replace them or stop machines
> without review by a safety engineer.

## Privacy by design

- Video frames stay in memory and are never written to disk by default.
- Workers are not identified: no face recognition, no names, no lasting IDs. Alerts refer to a location.
- Faces are blurred in anything shown or stored, in two layers: the head of every tracked person, plus
  every face a separate face detector finds (covers people the tracker misses, e.g. forklift drivers).
  Snapshots (if enabled) are deleted automatically.
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
python main.py --source video.mp4 --realtime   # play a file at real speed (like a live camera)
python tools/jev_check.py --source video.mp4 --frame 300   # one-off Jev test (see below)
python tools/jev_check.py --list-models    # which Jev models your API key can use
python tools/check_images.py --source data/images          # boxes + PPE crops for every photo
python tools/check_images.py --source data/images --jev    # ...and Jev's answers written on each person
python tools/calibrate.py --camera cam_floor_1 --source 0   # calibrate a camera (see below)
python -m pytest                           # tests
ruff check .                               # lint
```

## Docker (run locally)

```powershell
docker compose up --build      # first build takes several minutes
# open http://localhost:8000   (live view, faces blurred)
docker compose down            # stop

docker compose up --build images             # check every photo in data/images with Jev
# open http://localhost:8001   (annotated results, faces blurred; also saved in data/output)
docker compose --profile images down         # stop both services
```

- The video, camera and mode are set in `command:` in [docker-compose.yml](docker-compose.yml).
- The `images` service only starts when named, and every start sends 3 crops per person to Jev.
- `config/`, `calibration/` and `data/` are mounted read-only; `models/` and `logs/` are kept on your PC.
- `.env` (Jev API key) is passed in at run time and never copied into the image; neither are videos or site data.
- The port is published to this PC only (`127.0.0.1`). Use `"8000:8000"` to allow other PCs on the network.
- Webcams can't be used inside Docker on Windows. Use a video file, or a network camera (`rtsp://...`);
  for a quick test a phone IP-camera app works as a network camera.
- `tools/calibrate.py` and `tools/draw_machines.py` need a screen, so run them on the PC (not in Docker).
- Docker Desktop limits CPU and memory for containers; if it runs slowly, raise them in Docker Desktop settings
  (Resources) or in `%UserProfile%\.wslconfig`.

Without Docker, `python main.py --web 8000` gives the same browser view.

## CI

Every push to `main` and every pull request runs [.github/workflows/ci.yml](.github/workflows/ci.yml):
lint (ruff), unit tests (pytest), and a smoke test of the tracker on a synthetic video.
CI never uses real factory footage. A Docker image build and publish step (CD) is added with the Docker milestone.

On first run the pose model (`yolov8n-pose.pt`) and the face model (YuNet, checksum-verified) are
downloaded into `models/`, and the pose model is converted to OpenVINO format once. In the preview,
a green box marks each person and a blue box the area checked for the hard hat (it stops above the
eyebrows). A yellow box marks the shoes (found from the ankles); an orange box means the ankles were
hidden and the bottom of the person's box was used instead.

**Speed:** video files are analysed frame by frame, so on a slow PC they play in slow motion.
`--realtime` plays them at real speed and skips frames instead, exactly like a live camera.
Use 1080p or 720p streams where possible: 4K costs about twice as much CPU.
Set `alerts.expected_fps` to what your PC really achieves (printed at the end of a run); at startup
each machine's `danger_m` is checked against how far a fast walker (1.6 m/s, ISO 13855) moves
before an ALARM can fire.

## Camera calibration

Floor distances need one calibration per camera:

1. Mark at least 4 points on the floor with tape (5–6 is better), spread out, never 3 on one line.
2. Measure each point from one corner of the room: x to the right, y forward, in meters.
3. Run `python tools/calibrate.py --camera <camera id> --source <camera>`, click each point and type its position.
4. Check the reported error (aim for under 10 cm) and that the blue 1 m grid follows the floor, then press `s`.

The result is saved to `calibration/<camera>.json` (not committed: it is site-specific).

## Drawing machine areas

```powershell
python tools/draw_machines.py --camera <camera id> --source <camera or video> [--frame 120]
```

Click the corners **where each machine meets the floor** (not its top), press `n`, then answer the
questions in the terminal (id, name, warn/danger distance, noisy). Press `o` to add an operator spot,
`s` to save. The amber ring shows where WARN starts and the red ring where ALERT starts; the same rings
appear in `main.py --show`. Machines are saved to `config/machines.yaml` (not committed); machines of
other cameras in the file are kept, and the previous file is saved as `machines.yaml.bak`.

Testing with stock factory video (no real measurements): calibrate using objects of known size,
e.g. a EUR pallet (1.2 x 0.8 m) or floor tiles. Put clips in `data/videos/` (not committed).

## Proximity alerts

Each person near each running machine is tracked separately; the machine shows the highest level.

| Level | Starts when | What happens (default outputs) |
|---|---|---|
| WARN  | inside `warn_m` for 0.6 s | amber light |
| ALERT | inside `danger_m` for 1 s | red light, floor projection, operator screen, supervisor radio + announcement text |
| ALARM | inside half of `danger_m`, moving toward the machine, ALERT not acknowledged in 10 s, or still there 20 s after the announcement | flashing red, floor projection, operator, radio, voice (noisy machines: headset radio instead) |

- **Anti-flicker:** a level only drops after the person is 0.2 m past the line for 1 s.
- **Machine stopped:** no alerts. **Operator spot:** one person may stand there; stepping into the machine is never exempt.
- **Person lost from view** (e.g. behind the machine): the alert stays on for 3 s.
- **Announcements name the machine, never the person:** "Attention: person at Press 3, please step back 1.5 meters."
- **Never sound only:** every level must include a light or floor projection, because workers with ear defenders may not hear.
- All outputs are simulated for now (terminal + preview). Real lights/speakers/radios plug into `safety/outputs.py`.
- In the preview, press `a` to simulate the operator pressing "Announced".
- Incidents are logged as text to `logs/events.jsonl` (not committed): machine, times, peak level, closest distance,
  number of people, alarm reasons. No images, names or tracker numbers.

## Jev (PPE checks)

Each PPE item gets its own yes / no / unclear question to Jev, with its own crop, sent in parallel
from background threads so the video loop never waits for the network. The gate only unlocks
when every required item is a confident "yes"; "no", "unclear", errors and timeouts keep it locked.
The hard-hat crop stops above the eyes and the other crops come from the face-blurred frame.
Safety glasses are not sent to Jev; they will be checked locally.

**Trying Jev for the first time:** put `TYPESAFE_API_KEY=...` in `.env` (never in code or chat), then
`python tools/jev_check.py --source <video> --frame <n>`. It takes the largest person in that frame,
saves the 3 crops to `data/jev_check/` so you can see exactly what is sent, and makes one call per item
(no retries), printing the answer, confidence, time, tokens used and request id. Without a key, or with
`--dry-run`, it only saves the crops. Use stock or consented footage only.

**Checking photos:** `python tools/check_images.py --source data/images --jev` asks Jev about every
person in every photo and writes the answers on the annotated copy in `data/output/`, e.g.
`HAT yes 0.92` (green = confident yes, red = confident no, orange = unclear or below
`min_confidence`), followed by the gate result: ALL PPE OK, MISSING PPE or CHECK AGAIN.
Without `--jev` nothing is sent.

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
safety/alerts.py           WARN / ALERT / ALARM rules, anti-flicker, incidents
safety/outputs.py          lights, projector, voice, radios, operator screen (simulated adapters)
safety/events.py           text-only incident log
safety/layout.py           build / save machine layouts from clicked points, zone rings
tools/calibrate.py         interactive camera calibration
tools/draw_machines.py     click machine areas on the image -> config/machines.yaml
tools/jev_check.py         one-off Jev test with a single person's crops
tools/check_images.py      annotated photos with boxes, PPE crops and (with --jev) Jev's answers
safety/webview.py          browser preview (MJPEG) for Docker / no screen
Dockerfile, docker-compose.yml   local container setup
config/settings.yaml       general settings (committed)
config/machines.yaml       your factory layout (not committed; copy from the example)
models/                    model weights (not committed)
```

## Roadmap

1. Project skeleton
2. Pose tracking and ankle point (YOLOv8n-pose)
3. Camera calibration, floor distance, live-camera fix, background Jev inspector
4. Proximity alert levels (WARN / ALERT / ALARM), per-machine outputs, event log
5. Operator screen and event log
6. Gate: local safety-glasses check, test with real Jev API key
7. Docker (local) — CD (image published to GitHub Container Registry) still to do
