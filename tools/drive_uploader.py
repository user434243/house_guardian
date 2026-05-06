"""
tools/drive_uploader.py
Uploads video clips and metadata to Google Drive.
Uses OAuth2 service account credentials.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("drive")


class DriveUploader:
    def __init__(self, settings):
        self.cfg = settings.drive
        self._folder_id: Optional[str] = None
        self._service = None
        self._initialized = False

    def _init_service(self):
        if self._initialized:
            return
        creds_path = Path(self.cfg.credentials_file)
        if not creds_path.exists():
            log.warning(f"Drive credentials not found at {creds_path}. Uploads disabled.")
            return
        try:
            from googleapiclient.discovery import build
            from google.oauth2.service_account import Credentials

            creds = Credentials.from_service_account_file(
                str(creds_path),
                scopes=["https://www.googleapis.com/auth/drive.file"],
            )
            self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
            self._folder_id = self._get_or_create_folder()
            self._initialized = True
            log.info(f"Google Drive ready. Folder ID: {self._folder_id}")
        except ImportError:
            log.warning("google-api-python-client not installed. Drive uploads disabled.")
        except Exception as e:
            log.error(f"Drive init failed: {e}")

    def _get_or_create_folder(self) -> Optional[str]:
        results = (
            self._service.files()
            .list(
                q=f"name='{self.cfg.folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)",
            )
            .execute()
        )
        files = results.get("files", [])
        if files:
            return files[0]["id"]

        # Create folder
        meta = {
            "name": self.cfg.folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = self._service.files().create(body=meta, fields="id").execute()
        log.info(f"Created Drive folder: {self.cfg.folder_name}")
        return folder["id"]

    async def upload(self, clip_path: str, reason: str, scene) -> Optional[str]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._upload_sync, clip_path, reason, scene
        )

    def _upload_sync(self, clip_path: str, reason: str, scene) -> Optional[str]:
        self._init_service()
        if not self._service or not self._folder_id:
            log.warning("Drive not available, clip not uploaded.")
            return None

        p = Path(clip_path)
        if not p.exists():
            log.error(f"Clip file not found: {clip_path}")
            return None

        # Check size
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > self.cfg.max_clip_size_mb:
            log.warning(f"Clip too large ({size_mb:.1f} MB > {self.cfg.max_clip_size_mb} MB), skipping upload.")
            return None

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        description = (
            f"Trigger: {reason}\n"
            f"Time: {ts}\n"
            f"Anomaly score: {scene.anomaly_score:.2f}\n"
            f"Anomaly reasons: {'; '.join(scene.anomaly_reasons)}\n"
            f"Objects: {', '.join(set(d.label for d in scene.detections))}\n"
            f"Faces: {', '.join(f.identity for f in scene.faces)}\n"
            f"Plates: {', '.join(p.plate_text for p in scene.plates)}"
        )

        try:
            from googleapiclient.http import MediaFileUpload

            file_meta = {
                "name": p.name,
                "parents": [self._folder_id],
                "description": description,
            }
            media = MediaFileUpload(str(p), mimetype="video/mp4", resumable=True)
            uploaded = (
                self._service.files()
                .create(body=file_meta, media_body=media, fields="id,webViewLink")
                .execute()
            )
            link = uploaded.get("webViewLink", "")
            log.info(f"Uploaded to Drive: {p.name} → {link}")

            # Also upload metadata as a JSON sidecar
            meta_path = p.with_suffix(".json")
            meta_path.write_text(json.dumps({
                "clip": p.name,
                "reason": reason,
                "timestamp": ts,
                "anomaly_score": scene.anomaly_score,
                "anomaly_reasons": scene.anomaly_reasons,
                "detections": [{"label": d.label, "confidence": d.confidence} for d in scene.detections],
                "faces": [{"identity": f.identity, "confidence": f.confidence} for f in scene.faces],
                "plates": [{"plate": p.plate_text, "trusted": p.trusted} for p in scene.plates],
                "drive_link": link,
            }, indent=2))

            json_meta = {
                "name": meta_path.name,
                "parents": [self._folder_id],
            }
            json_media = MediaFileUpload(str(meta_path), mimetype="application/json")
            self._service.files().create(body=json_meta, media_body=json_media).execute()

            return link
        except Exception as e:
            log.error(f"Drive upload failed: {e}")
            return None
