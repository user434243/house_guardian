"""
tools/servo.py
Pan/tilt servo controller using Raspberry Pi GPIO PWM.
Falls back to a stub when not running on Pi hardware.
"""
from __future__ import annotations
import asyncio
import logging

log = logging.getLogger("servo")

# Servo PWM constants (standard hobby servos)
SERVO_FREQ_HZ = 50
DUTY_MIN = 2.5    # 0.5ms pulse → -90°
DUTY_MID = 7.5    # 1.5ms pulse → 0°
DUTY_MAX = 12.5   # 2.5ms pulse → +90°


def _angle_to_duty(angle: float, min_angle: float = -90, max_angle: float = 90) -> float:
    """Convert angle in degrees to PWM duty cycle percentage."""
    clamped = max(min_angle, min(max_angle, angle))
    normalized = (clamped - min_angle) / (max_angle - min_angle)  # 0.0 to 1.0
    return DUTY_MIN + normalized * (DUTY_MAX - DUTY_MIN)


class ServoController:
    def __init__(self, pan_pin: int, tilt_pin: int):
        self._pan_pin = pan_pin
        self._tilt_pin = tilt_pin
        self._pan_angle = 0.0
        self._tilt_angle = 0.0
        self._gpio_available = False
        self._pan_pwm = None
        self._tilt_pwm = None

        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(pan_pin, GPIO.OUT)
            GPIO.setup(tilt_pin, GPIO.OUT)
            self._pan_pwm = GPIO.PWM(pan_pin, SERVO_FREQ_HZ)
            self._tilt_pwm = GPIO.PWM(tilt_pin, SERVO_FREQ_HZ)
            self._pan_pwm.start(DUTY_MID)
            self._tilt_pwm.start(DUTY_MID)
            self._gpio = GPIO
            self._gpio_available = True
            log.info(f"Servo GPIO initialized: pan=GPIO{pan_pin}, tilt=GPIO{tilt_pin}")
        except (ImportError, RuntimeError) as e:
            log.warning(f"GPIO not available ({e}), servo running in simulation mode.")

    async def move(self, pan_degrees: float, tilt_degrees: float):
        """Move camera to absolute pan/tilt angles (in degrees)."""
        # Clamp
        pan_degrees = max(-90, min(90, pan_degrees))
        tilt_degrees = max(-30, min(60, tilt_degrees))

        self._pan_angle = pan_degrees
        self._tilt_angle = tilt_degrees

        if self._gpio_available:
            pan_duty = _angle_to_duty(pan_degrees, -90, 90)
            tilt_duty = _angle_to_duty(tilt_degrees, -30, 60)
            await asyncio.get_event_loop().run_in_executor(
                None, self._apply_pwm, pan_duty, tilt_duty
            )
        else:
            log.debug(f"[SIM] Servo move → pan={pan_degrees:.1f}° tilt={tilt_degrees:.1f}°")

        # Small delay for servo to reach position
        await asyncio.sleep(0.3)

    def _apply_pwm(self, pan_duty: float, tilt_duty: float):
        if self._pan_pwm:
            self._pan_pwm.ChangeDutyCycle(pan_duty)
        if self._tilt_pwm:
            self._tilt_pwm.ChangeDutyCycle(tilt_duty)
        import time
        time.sleep(0.3)
        # Stop sending PWM pulses to reduce servo jitter/heat
        if self._pan_pwm:
            self._pan_pwm.ChangeDutyCycle(0)
        if self._tilt_pwm:
            self._tilt_pwm.ChangeDutyCycle(0)

    async def center(self):
        """Move to center position."""
        await self.move(0, 0)
        log.info("Servo centered.")

    async def park(self):
        """Move to safe parked position and stop PWM."""
        await self.center()
        if self._gpio_available:
            if self._pan_pwm:
                self._pan_pwm.stop()
            if self._tilt_pwm:
                self._tilt_pwm.stop()
            self._gpio.cleanup()
        log.info("Servo parked.")

    @property
    def position(self) -> tuple:
        return (self._pan_angle, self._tilt_angle)
