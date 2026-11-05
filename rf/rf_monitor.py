"""
Optional receive-only RF Monitor – Stage 6

Responsibilities:
- Detect presence of an SDR (RTL-SDR class or wider-band via SoapySDR)
- Report observed frequency, signal strength, bandwidth when available
- Generate simple spectrum snapshot data for the dashboard

IMPORTANT SAFETY RULES:
- RECEIVE-ONLY
- Never transmits
- No jamming, spoofing, interference, command injection, or takeover
- If no SDR is present → status = "RF SENSOR OFFLINE"
- Never fabricates RF values
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try optional SDR backends (none are required)
_RTLSDR = False
_SOAPY = False

try:
    from rtlsdr import RtlSdr
    _RTLSDR = True
except ImportError:
    pass

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    _SOAPY = True
except ImportError:
    pass


class RFMonitor:
    """
    Optional receive-only RF observer.

    When no SDR hardware or driver is available the module stays offline
    and the rest of the system continues normally.
    """

    def __init__(
        self,
        enabled: bool = True,
        device: str = "auto",          # "auto", "rtlsdr", "soapy", or specific device arg
        center_freq_hz: float = 2.4e9,
        sample_rate: float = 2.4e6,
        gain: str | float = "auto",
        poll_interval_s: float = 1.0,
    ):
        self.enabled = enabled
        self.device_pref = device
        self.center_freq_hz = center_freq_hz
        self.sample_rate = sample_rate
        self.gain = gain
        self.poll_interval_s = poll_interval_s

        self.sdr = None
        self.backend: Optional[str] = None
        self.status = "RF SENSOR OFFLINE"
        self.last_observation: Dict[str, Any] = self._offline_result()
        self._last_poll = 0.0

        if not enabled:
            logger.info("RF monitor disabled by configuration")
            return

        self._try_connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _try_connect(self):
        """Attempt to open an SDR. Failures leave the module offline."""
        if self.device_pref in ("auto", "rtlsdr") and _RTLSDR:
            if self._open_rtlsdr():
                return

        if self.device_pref in ("auto", "soapy") and _SOAPY:
            if self._open_soapy():
                return

        # No usable backend / hardware
        self.status = "RF SENSOR OFFLINE"
        self.sdr = None
        self.backend = None
        logger.warning(
            "RF SENSOR OFFLINE – no compatible SDR found. "
            "Install pyrtlsdr / SoapySDR and connect a receive-only SDR "
            "to enable RF monitoring. System continues without RF."
        )

    def _open_rtlsdr(self) -> bool:
        try:
            sdr = RtlSdr()
            sdr.sample_rate = self.sample_rate
            sdr.center_freq = self.center_freq_hz
            if self.gain == "auto":
                sdr.gain = "auto"
            else:
                sdr.gain = float(self.gain)
            self.sdr = sdr
            self.backend = "rtlsdr"
            self.status = "ok"
            logger.info(
                f"RF monitor ONLINE (RTL-SDR) | "
                f"center={self.center_freq_hz/1e6:.3f} MHz | "
                f"rate={self.sample_rate/1e6:.2f} MS/s"
            )
            return True
        except Exception as e:
            logger.debug(f"RTL-SDR open failed: {e}")
            return False

    def _open_soapy(self) -> bool:
        try:
            results = SoapySDR.Device.enumerate()
            if not results:
                return False
            # Use first available device
            sdr = SoapySDR.Device(dict(results[0]))
            sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq_hz)
            if self.gain != "auto":
                try:
                    sdr.setGain(SOAPY_SDR_RX, 0, float(self.gain))
                except Exception:
                    pass
            self.sdr = sdr
            self.backend = "soapy"
            self.status = "ok"
            logger.info(
                f"RF monitor ONLINE (SoapySDR) | "
                f"center={self.center_freq_hz/1e6:.3f} MHz"
            )
            return True
        except Exception as e:
            logger.debug(f"SoapySDR open failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _offline_result(self) -> Dict[str, Any]:
        return {
            "detected": False,
            "status": "RF SENSOR OFFLINE",
            "frequency_hz": None,
            "frequency_mhz": None,
            "signal_strength_dbm": None,
            "bandwidth_hz": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": None,
            "spectrum": None,
        }

    def poll(self, force: bool = False) -> Dict[str, Any]:
        """
        Read a short burst of samples and produce an observation.
        Rate-limited by poll_interval_s unless force=True.
        """
        now = time.time()
        if not force and (now - self._last_poll) < self.poll_interval_s:
            return self.last_observation

        self._last_poll = now

        if not self.enabled or self.sdr is None or self.status != "ok":
            self.last_observation = self._offline_result()
            return self.last_observation

        try:
            if self.backend == "rtlsdr":
                obs = self._poll_rtlsdr()
            elif self.backend == "soapy":
                obs = self._poll_soapy()
            else:
                obs = self._offline_result()

            self.last_observation = obs
            return obs
        except Exception as e:
            logger.warning(f"RF poll failed: {e} – marking offline")
            self.status = "RF SENSOR OFFLINE"
            self.last_observation = self._offline_result()
            # Attempt clean close
            self.close()
            return self.last_observation

    def _poll_rtlsdr(self) -> Dict[str, Any]:
        """Grab a few samples and compute a simple power estimate."""
        import numpy as np

        samples = self.sdr.read_samples(256 * 1024)
        # Power in linear scale → approximate dB
        power = np.mean(np.abs(samples) ** 2)
        # Very rough conversion to dBm (not calibrated – still real measured value)
        power_db = 10 * np.log10(power + 1e-12)
        # Map to a plausible dBm range for display (still derived from real samples)
        signal_dbm = round(float(power_db - 30), 1)  # arbitrary offset, real relative power

        # Simple peak detection: if power is significantly above noise floor
        noise_est = np.median(np.abs(samples) ** 2)
        detected = bool(power > noise_est * 4)

        # Spectrum snapshot (downsample for dashboard)
        spectrum = self._simple_spectrum(samples)

        return {
            "detected": detected,
            "status": "ok",
            "frequency_hz": float(self.center_freq_hz),
            "frequency_mhz": round(self.center_freq_hz / 1e6, 3),
            "signal_strength_dbm": signal_dbm,
            "bandwidth_hz": float(self.sample_rate),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": "rtlsdr",
            "spectrum": spectrum,
        }

    def _poll_soapy(self) -> Dict[str, Any]:
        """Minimal SoapySDR read – returns basic info when stream works."""
        # Full streaming setup is more involved; for Stage 6 we report
        # that the device is present and the configured frequency.
        # Real spectrum can be expanded later.
        return {
            "detected": True,  # device is open and tuned
            "status": "ok",
            "frequency_hz": float(self.center_freq_hz),
            "frequency_mhz": round(self.center_freq_hz / 1e6, 3),
            "signal_strength_dbm": None,  # not computed yet
            "bandwidth_hz": float(self.sample_rate),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": "soapy",
            "spectrum": None,
        }

    def _simple_spectrum(self, samples, n_fft: int = 256, n_points: int = 64) -> List[float]:
        """Return a short magnitude spectrum for dashboard plotting."""
        import numpy as np
        # Use a subset
        x = samples[:n_fft]
        if len(x) < n_fft:
            return []
        window = np.hanning(len(x))
        spec = np.abs(np.fft.rfft(x * window))
        # Downsample
        step = max(1, len(spec) // n_points)
        down = spec[::step][:n_points]
        # Log scale
        down = 20 * np.log10(down + 1e-12)
        return [round(float(v), 1) for v in down]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_latest(self) -> Dict[str, Any]:
        """Return the most recent observation (polls if needed)."""
        return self.poll()

    def is_online(self) -> bool:
        return self.status == "ok" and self.sdr is not None

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "backend": self.backend,
            "center_freq_mhz": round(self.center_freq_hz / 1e6, 3) if self.center_freq_hz else None,
            "libraries": {
                "pyrtlsdr": _RTLSDR,
                "SoapySDR": _SOAPY,
            },
        }

    def close(self):
        """Release SDR resources."""
        if self.sdr is not None:
            try:
                if self.backend == "rtlsdr":
                    self.sdr.close()
                # SoapySDR Device has no universal close; just drop reference
            except Exception:
                pass
            self.sdr = None
            self.backend = None
            self.status = "RF SENSOR OFFLINE"
            logger.info("RF monitor closed")
