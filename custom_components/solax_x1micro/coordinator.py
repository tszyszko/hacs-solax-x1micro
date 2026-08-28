"""Data coordinator for the SolaX X1-Micro integration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from homeassistant.components import mqtt
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util

from .const import CONF_SERIAL_NUMBER, MQTT_TOPIC_DATA, MQTT_TOPIC_STATUS
from .frame_decoder import (
    decode_solax_event_frame,
    decode_solax_frame,
    is_valid_solax_frame,
)

_LOGGER = logging.getLogger(__name__)


class SolaxCoordinator:
    """Manages MQTT subscriptions and holds the latest inverter data."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self.serial_number: str = entry.data[CONF_SERIAL_NUMBER]
        self.data: dict[str, Any] = {}
        self.online: bool = False
        self._listeners: list[Callable[[], None]] = []
        self._unsub_data: Callable | None = None
        self._unsub_status: Callable | None = None
        self._frames_ok: int = 0
        self._frames_error: int = 0

    async def async_setup(self) -> None:
        """Subscribe to MQTT topics."""
        topic_data = MQTT_TOPIC_DATA.format(self.serial_number)
        topic_status = MQTT_TOPIC_STATUS.format(self.serial_number)

        self._unsub_data = await mqtt.async_subscribe(
            self.hass,
            topic_data,
            self._on_data_message,
            encoding=None,  # receive raw bytes
        )
        self._unsub_status = await mqtt.async_subscribe(
            self.hass,
            topic_status,
            self._on_status_message,
        )
        _LOGGER.debug("Subscribed to MQTT topics: %s, %s", topic_data, topic_status)

    @callback
    def _on_data_message(self, msg: mqtt.ReceiveMessage) -> None:
        """Handle incoming binary data frame."""
        payload: bytes = msg.payload
        parsed = decode_solax_frame(payload)
        if parsed is None:
            event = decode_solax_event_frame(payload)
            if event is not None:
                # Transient fault / status report: keep the last measurements,
                # update the run mode and remember the codes.
                self._frames_ok += 1
                self.data = {
                    **self.data,
                    "run_mode": event["run_mode"],
                    "event_codes": event["event_codes"],
                    "event_time": dt_util.utcnow(),
                    "frames_ok": self._frames_ok,
                    "frames_error": self._frames_error,
                }
                self.online = True
                _LOGGER.debug("Decoded SolaX event frame: %s", event)
                self._notify_listeners()
                return
            if is_valid_solax_frame(payload):
                # Well-formed but not a frame type we decode (boot sequence,
                # config dump, ...).  Not an error; do not touch the counters.
                _LOGGER.debug(
                    "Ignoring valid but unsupported frame on %s (len=%d, func=0x%02X)",
                    msg.topic,
                    len(payload),
                    payload[7] if len(payload) > 7 else 0,
                )
                return
            self._frames_error += 1
            _LOGGER.warning(
                "Received invalid or unrecognized frame on %s (len=%d)",
                msg.topic,
                len(payload),
            )
            self.data = {
                **self.data,
                "frames_ok": self._frames_ok,
                "frames_error": self._frames_error,
            }
            self._notify_listeners()
            return
        self._frames_ok += 1
        self.data = {
            "event_codes": self.data.get("event_codes"),
            "event_time": self.data.get("event_time"),
            **parsed,
            "frames_ok": self._frames_ok,
            "frames_error": self._frames_error,
        }
        self.online = True
        _LOGGER.debug("Decoded SolaX frame: %s", parsed)
        self._notify_listeners()

    @callback
    def _on_status_message(self, msg: mqtt.ReceiveMessage) -> None:
        """Handle 'hello mqtt!' keepalive message."""
        _LOGGER.debug("SolaX status heartbeat on %s: %s", msg.topic, msg.payload)
        self.online = True
        self._notify_listeners()

    def async_add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Register a listener that is called on data updates.

        Returns a callable that removes the listener.
        """
        self._listeners.append(listener)

        def remove_listener() -> None:
            self._listeners.remove(listener)

        return remove_listener

    def _notify_listeners(self) -> None:
        for listener in list(self._listeners):
            listener()

    @callback
    def async_unload(self) -> None:
        """Unsubscribe from MQTT topics."""
        if self._unsub_data is not None:
            self._unsub_data()
            self._unsub_data = None
        if self._unsub_status is not None:
            self._unsub_status()
            self._unsub_status = None
