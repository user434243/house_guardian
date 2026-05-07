"""
tools/audio_monitor.py
Listens on a USB microphone and classifies sounds in real time.
Runs as a background asyncio task alongside the vision pipeline.

Sound classes detected:
  glass_break  — high priority, immediate alert
  shout/scream — high priority
  dog_bark     — medium (could be intruder deterring a dog)
  vehicle      — low (engine, door slam)
  doorbell     — informational
  ambient      — background noise, no action
  silence      — nothing

Uses yamnet-class sound classification via TensorFlow Lite
if available, otherwise falls back to energy + zero-crossing
heuristic classifier (works on Pi without GPU).
"""
from __future__ import annotations

import asyncio
import logging
import time
import wave
import struct
from pathlib import Path
from typing import Optional

log = logging.getLogger("audio")

SAMPLE_RATE    = 16000    # Hz — YAMNet requirement
CHUNK_FRAMES   = 1024     # frames per PyAudio read
WINDOW_SEC     = 1.0      # seconds of audio per classification pass
ALERT_CLASSES  = {"glass_break", "shout", "scream", "gunshot"}
MEDIUM_CLASSES = {"dog_bark", "alarm", "siren"}


class AudioMonitor:
    def __init__(self, settings, event_logger, alerter):
        self.settings  = settings
        self.db        = event_logger
        self.alerter   = alerter
        self._running  = False
        self._yamnet   = None
        self._use_yamnet = False
        self._last_alert_at: dict[str, float] = {}
        self._alert_cooldown = 60   # seconds between same-class alerts

        self._try_load_yamnet()

    def _try_load_yamnet(self):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            self._yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            self._use_yamnet = True
            log.info("YAMNet sound classifier loaded.")
        except Exception as e:
            log.warning(f"YAMNet not available ({e}), using heuristic classifier.")

    async def run(self):
        """Main audio loop. Run as asyncio task."""
        try:
            import pyaudio
        except ImportError:
            log.warning("PyAudio not installed. Audio monitoring disabled.")
            log.warning("Install with: pip install pyaudio --break-system-packages")
            return

        pa = pyaudio.PyAudio()
        device_index = self._find_usb_mic(pa)
        if device_index is None:
            log.warning("No USB microphone found. Audio monitoring disabled.")
            pa.terminate()
            return

        log.info(f"Audio monitor starting on device index {device_index}.")
        self._running = True

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_FRAMES,
        )

        buffer = []
        frames_per_window = int(SAMPLE_RATE * WINDOW_SEC)

        try:
            while self._running:
                # Read non-blocking
                try:
                    data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                    buffer.extend(struct.unpack(f"{CHUNK_FRAMES}h", data))
                except Exception as e:
                    log.debug(f"Audio read error: {e}")
                    await asyncio.sleep(0.05)
                    continue

                if len(buffer) >= frames_per_window:
                    window = buffer[:frames_per_window]
                    buffer = buffer[frames_per_window:]

                    # Run classification in thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self._classify, window
                    )

                    if result:
                        await self._handle_detection(result)

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            log.info("Audio monitor stopped.")

    def _find_usb_mic(self, pa) -> Optional[int]:
        """Find the first USB audio input device."""
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                name = info["name"].lower()
                if "usb" in name or "microphone" in name or "mic" in name:
                    log.info(f"Found mic: [{i}] {info['name']}")
                    return i
        # Fall back to default input
        try:
            idx = pa.get_default_input_device_info()["index"]
            log.info(f"Using default input device: {idx}")
            return idx
        except Exception:
            return None

    def _classify(self, samples: list) -> Optional[dict]:
        """Classify a window of audio samples. Returns result dict or None."""
        import numpy as np

        audio = np.array(samples, dtype=np.float32) / 32768.0   # normalize to [-1, 1]
        db_level = float(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9))

        # Silence gate
        if db_level < -40:
            return None

        if self._use_yamnet:
            return self._classify_yamnet(audio, db_level)
        else:
            return self._classify_heuristic(audio, db_level)

    def _classify_yamnet(self, audio, db_level: float) -> Optional[dict]:
        """YAMNet classification."""
        import numpy as np
        try:
            scores, embeddings, spectrogram = self._yamnet(audio)
            scores_np = scores.numpy()
            mean_scores = scores_np.mean(axis=0)
            top_idx = mean_scores.argmax()
            confidence = float(mean_scores[top_idx])

            # Map YAMNet class indices to our simplified labels
            yamnet_class = self._yamnet_idx_to_label(top_idx)

            if confidence < 0.3:
                return None

            return {
                "sound_class": yamnet_class,
                "confidence": confidence,
                "db_level": db_level,
                "duration_sec": WINDOW_SEC,
            }
        except Exception as e:
            log.debug(f"YAMNet error: {e}")
            return None

    def _yamnet_idx_to_label(self, idx: int) -> str:
        """Map YAMNet class index to our simplified security labels."""
        # Key YAMNet indices (from yamnet class map)
        mapping = {
            # Glass / breaking
            **{i: "glass_break" for i in range(81, 83)},
            # Gunshot
            **{i: "gunshot" for i in [427, 428, 429]},
            # Shout / scream
            **{i: "shout" for i in [26, 27, 28, 29]},
            # Dog
            **{i: "dog_bark" for i in [74, 75, 76, 77]},
            # Alarm
            **{i: "alarm" for i in [388, 389, 390, 391, 392]},
            # Vehicle
            **{i: "vehicle" for i in [300, 301, 302, 303, 304, 305]},
        }
        return mapping.get(idx, "ambient")

    def _classify_heuristic(self, audio, db_level: float) -> Optional[dict]:
        """
        Fallback heuristic classifier using:
        - Zero-crossing rate (ZCR) — high for glass break
        - RMS energy — high for shouts
        - Spectral centroid proxy — distinguishes bark from engine
        """
        import numpy as np

        n = len(audio)
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Zero-crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * n))

        # Rough spectral centroid via FFT
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        spectral_centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-9))

        # Classification rules
        if zcr > 0.3 and rms > 0.1:
            return {"sound_class": "glass_break", "confidence": 0.65,
                    "db_level": db_level, "duration_sec": WINDOW_SEC}
        elif rms > 0.15 and spectral_centroid > 1000:
            return {"sound_class": "shout", "confidence": 0.60,
                    "db_level": db_level, "duration_sec": WINDOW_SEC}
        elif 500 < spectral_centroid < 2000 and zcr > 0.1:
            return {"sound_class": "dog_bark", "confidence": 0.55,
                    "db_level": db_level, "duration_sec": WINDOW_SEC}
        elif spectral_centroid < 500 and rms > 0.05:
            return {"sound_class": "vehicle", "confidence": 0.50,
                    "db_level": db_level, "duration_sec": WINDOW_SEC}
        elif rms > 0.03:
            return {"sound_class": "ambient", "confidence": 0.70,
                    "db_level": db_level, "duration_sec": WINDOW_SEC}

        return None

    def _save_audio_clip(self, samples: list) -> str:
        """Save a WAV clip for evidence."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"data/clips/audio_{ts}.wav"
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(struct.pack(f"{len(samples)}h", *samples))
        return path

    async def _handle_detection(self, result: dict):
        """React to a classified sound event."""
        sound_class = result["sound_class"]
        confidence  = result["confidence"]
        db_level    = result["db_level"]

        # Always log to DB
        await self.db.log_audio(
            sound_class=sound_class,
            confidence=confidence,
            db_level=db_level,
            duration_sec=result["duration_sec"],
        )

        if sound_class == "ambient":
            return

        log.info(f"Audio event: {sound_class} (conf={confidence:.2f}, dB={db_level:.1f})")

        # Check cooldown
        now = time.time()
        if now - self._last_alert_at.get(sound_class, 0) < self._alert_cooldown:
            return

        # Send alert for dangerous sounds
        if sound_class in ALERT_CLASSES:
            self._last_alert_at[sound_class] = now
            await self.alerter.send(
                subject=f"🔊 Audio Alert: {sound_class.replace('_', ' ').title()} Detected",
                body=(
                    f"Dangerous sound detected by House Guardian microphone.\n\n"
                    f"Sound: {sound_class.replace('_', ' ').upper()}\n"
                    f"Confidence: {confidence:.0%}\n"
                    f"Volume: {db_level:.0f} dB\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"Check your camera feed and recorded clips immediately."
                ),
                severity="high",
            )
        elif sound_class in MEDIUM_CLASSES:
            self._last_alert_at[sound_class] = now
            await self.alerter.send(
                subject=f"🔊 Audio Notice: {sound_class.replace('_', ' ').title()}",
                body=(
                    f"Sound detected: {sound_class}\n"
                    f"Confidence: {confidence:.0%} | Volume: {db_level:.0f} dB\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                ),
                severity="medium",
            )
