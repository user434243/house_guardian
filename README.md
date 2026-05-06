# 🏠 House Guardian — Autonomous AI Security Agent

A fully autonomous home security system running **locally on a Raspberry Pi 5**. No cloud subscription required. The agent thinks for itself — it decides where to point the camera, detects threats, and alerts you by email, all powered by a local LLM.

---

## System Architecture

```
Camera (Picamera2 HQ)
    ↓
Vision Pipeline
  ├── YOLO v8 — object detection + tracking (person, car, truck, bag...)
  ├── YOLO Pose — skeleton / behavior analysis (loitering, running, crouching)
  ├── DeepFace — face recognition against your known_faces database
  └── EasyOCR — license plate reading
    ↓
SceneState (fused snapshot every N frames)
    ↓
Local LLM Agent (Ollama + Llama3 / Phi-3)
  ├── Reads scene in natural language
  ├── Reasons about threats and behaviors
  └── Calls tools:
      ├── move_camera(pan, tilt)    → GPIO PWM → Servo
      ├── send_alert(subject, body) → SMTP → Your email
      ├── start_recording()         → MP4 clip → Google Drive
      ├── log_event()               → SQLite database
      └── patrol_sweep()            → 180° automated patrol
```

---

## Hardware Required

| Component | Notes |
|-----------|-------|
| Raspberry Pi 5 (4GB or 8GB) | 8GB recommended for larger LLM models |
| Hailo-8L Hat2 AI accelerator | Offloads YOLO inference, dramatically speeds up vision |
| Raspberry Pi HQ Camera | Or Camera Module 3 with zoom lens |
| Pan/tilt servo bracket | Standard 2-axis servo mount |
| 2× SG90 or MG90S servos | One for pan, one for tilt |
| microSD 64GB+ (A2 rated) | Fast I/O matters |
| Heatsink + fan for Pi 5 | Active cooling required under load |

### GPIO Wiring
```
Pan servo  → GPIO18 (BCM) + 5V + GND
Tilt servo → GPIO12 (BCM) + 5V + GND

⚠️  Power servos from a separate 5V source, not the Pi's 5V pin.
    Heavy servo load can brown out the Pi.
```

---

## Quick Start

```bash
# 1. Clone / copy the project to your Pi
git clone <repo> house_guardian && cd house_guardian

# 2. Run setup (installs deps, pulls LLM model)
chmod +x setup.sh && ./setup.sh

# 3. Edit config
nano config/config.yaml

# 4. Add known faces
# Create a folder per person with 3-5 photos:
mkdir -p data/known_faces/John
cp john_photo1.jpg john_photo2.jpg data/known_faces/John/

# 5. Run
python3 main.py
```

---

## Configuration

Edit `config/config.yaml`:

```yaml
llm:
  model: "llama3.2:3b"           # Fast for Pi 5 + Hat2
  reasoning_interval_sec: 8.0   # How often the agent thinks

alert:
  smtp_user: "yourgmail@gmail.com"
  smtp_password: "your_app_password"   # Gmail App Password
  recipient_email: "you@example.com"

knowledge:
  owner_name: "Your Name"
  trusted_vehicle_plates:
    - "ABC123"
  normal_hours: "07:00 - 22:00"
```

---

## LLM Model Recommendations

| Model | RAM | Speed | Quality |
|-------|-----|-------|---------|
| `phi3:mini` | 2.3GB | Very fast | Good |
| `llama3.2:3b` | 2.0GB | Fast ✅ Default | Good |
| `llama3.1:8b` | 4.7GB | Moderate | Better |
| `mistral:7b` | 4.1GB | Moderate | Good |

Pull any model: `ollama pull MODEL_NAME`

---

## Google Drive Setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable **Google Drive API**
3. Create a **Service Account** → Download JSON key
4. Save as `config/drive_credentials.json`
5. In Google Drive, create a folder named `HouseGuardian`
6. Share that folder with the service account email address

---

## Known Faces Database

```
data/
  known_faces/
    John/
      photo1.jpg
      photo2.jpg
      photo3.jpg
    Sarah/
      photo1.jpg
      ...
```

Each subfolder name becomes the person's identity in alerts.
Use clear frontal face photos in good lighting. 3-5 photos per person is enough.

---

## Agent Behavior

The agent runs a reasoning cycle every 8 seconds (configurable). Each cycle it:

1. **Reads the scene** — objects, faces, plates, poses, anomaly score
2. **Reasons** with the LLM about what's happening
3. **Acts** by calling one or more tools:
   - **Quiet scene** → patrol sweep or move to check a zone
   - **Person detected** → track them, assess behavior
   - **Unknown face** → alert + record
   - **Untrusted vehicle idling** → alert + record
   - **Anomaly score > 0.5** → always alert + record
   - **Outside normal hours + person** → alert

The agent has full knowledge of:
- Your trusted vehicle plates
- Known faces and who they are
- Normal activity hours
- Named camera zones and their angles
- Suspicious behavior patterns

---

## Project Structure

```
house_guardian/
├── main.py                  ← Entry point
├── setup.sh                 ← One-time setup
├── requirements.txt
├── config/
│   ├── config.yaml          ← Your configuration
│   └── drive_credentials.json
├── agent/
│   └── guardian.py          ← LLM agent + tool calling loop
├── vision/
│   └── pipeline.py          ← Camera, YOLO, faces, plates, poses
├── tools/
│   ├── servo.py             ← GPIO servo controller
│   ├── alerter.py           ← Email alerts
│   ├── recorder.py          ← Video clip recording
│   ├── drive_uploader.py    ← Google Drive upload
│   └── event_logger.py      ← SQLite event log
└── data/
    ├── clips/               ← Recorded video clips
    ├── snapshots/           ← Alert snapshots
    ├── logs/                ← events.db + guardian.log
    └── known_faces/         ← Face recognition database
```

---

## Run as System Service

```bash
sudo nano /etc/systemd/system/guardian.service
```

```ini
[Unit]
Description=House Guardian Security Agent
After=network.target ollama.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/house_guardian
ExecStart=/usr/bin/python3 main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable guardian
sudo systemctl start guardian
sudo journalctl -fu guardian   # follow logs
```

---

## Viewing Event History

```python
from tools.event_logger import EventLogger
log = EventLogger()

# Last 24 hours of events
events = log.query_recent(hours=24)
for e in events:
    print(e)

# Only suspicious behavior events
alerts = log.query_recent(hours=72, event_type="suspicious_behavior")
```

---

## License
MIT — build freely, stay safe. 🔒
