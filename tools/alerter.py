"""
tools/alerter.py
Sends email alerts to the owner with optional snapshot attachment.
"""
from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Optional

log = logging.getLogger("alerter")

SEVERITY_EMOJI = {
    "low": "🟡",
    "medium": "🟠",
    "high": "🔴",
}


class Alerter:
    def __init__(self, settings):
        self.cfg = settings.alert
        self.owner = settings.knowledge.owner_name

    async def send(
        self,
        subject: str,
        body: str,
        severity: str = "medium",
        snapshot_path: Optional[str] = None,
    ) -> bool:
        if not self.cfg.smtp_user or not self.cfg.recipient_email:
            log.warning("Email not configured — alert not sent.")
            return False

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._send_sync, subject, body, severity, snapshot_path
            )
            return result
        except Exception as e:
            log.error(f"Failed to send alert: {e}")
            return False

    def _send_sync(
        self,
        subject: str,
        body: str,
        severity: str,
        snapshot_path: Optional[str],
    ) -> bool:
        emoji = SEVERITY_EMOJI.get(severity, "⚠️")
        full_subject = f"{emoji} House Guardian [{severity.upper()}]: {subject}"

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        html_body = f"""
<html><body style="font-family: Arial, sans-serif; color: #222;">
<h2 style="color: {'#c0392b' if severity=='high' else '#e67e22' if severity=='medium' else '#f1c40f'}">
  {emoji} House Guardian Security Alert
</h2>
<p><strong>Severity:</strong> {severity.upper()}</p>
<p><strong>Time:</strong> {ts}</p>
<hr/>
<p>{body.replace(chr(10), "<br/>")}</p>
<hr/>
<p style="color: #888; font-size: 12px;">
  This alert was generated automatically by House Guardian on your Raspberry Pi.<br/>
  Video clips are being uploaded to Google Drive in the HouseGuardian folder.
</p>
</body></html>
"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = full_subject
        msg["From"] = f"House Guardian <{self.cfg.smtp_user}>"
        msg["To"] = self.cfg.recipient_email

        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        if snapshot_path and Path(snapshot_path).exists():
            with open(snapshot_path, "rb") as f:
                img_data = f.read()
            img = MIMEImage(img_data, name=Path(snapshot_path).name)
            img.add_header("Content-Disposition", "attachment", filename=Path(snapshot_path).name)
            msg.attach(img)

        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(self.cfg.smtp_user, self.cfg.smtp_password)
            server.sendmail(self.cfg.smtp_user, self.cfg.recipient_email, msg.as_string())

        log.info(f"Alert email sent to {self.cfg.recipient_email}")
        return True
