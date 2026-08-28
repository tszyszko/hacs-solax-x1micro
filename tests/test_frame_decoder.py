"""Regression tests for the ``$$`` real-time frame decoder.

The decoder module is loaded by file path so the tests run with a bare Python
interpreter (no Home Assistant install needed):

    python -m unittest discover -s tests
"""

from __future__ import annotations

import importlib.util
import struct
import unittest
from pathlib import Path

_DECODER = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "solax_x1micro"
    / "frame_decoder.py"
)
_spec = importlib.util.spec_from_file_location("frame_decoder", _DECODER)
assert _spec is not None and _spec.loader is not None
frame_decoder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frame_decoder)

crc16_buypass = frame_decoder.crc16_buypass
decode_solax_frame = frame_decoder.decode_solax_frame
decode_solax_event_frame = frame_decoder.decode_solax_event_frame
is_valid_solax_frame = frame_decoder.is_valid_solax_frame

# Real 107-byte real-time frame captured from an X1-Micro 2 in 1 running
# WiFi firmware 004.06 (loc/tsp/<sn>, 2026-08-28).  Bytes 0x3C-0x3D are
# ``04 06`` - the DSP firmware version, not the 0x0205 "marker" seen on
# 005.02 units.
FRAME_FW_004_06 = bytes.fromhex(
    "24246b000801011c33304d335931303130513036363400000000000000010e4800"
    "0202190033304d335931303130513036363400000000000000b00404060128007b"
    "0904006e007c13b501ab010f0010004200460002000f0000000f00210021000300"
    "0000000000002be3"
)

# 46-byte boot frame (function code 0x0E) from the same unit; carries the
# WiFi firmware string "004.06" and must be rejected by the real-time decoder.
FRAME_BOOT_FW_STRING = bytes.fromhex(
    "24242e000801020e33304d335931303130513036363400000000000000010e0b00"
    "010207000e3030342e303615a8"
)

# Event frames (run_mode 3) captured while the AC side was unplugged (first)
# and plugged back in (second and third), 2026-08-28 17:32-17:33.
FRAME_EVENT_RAISED = bytes.fromhex(
    "242464000801011c33304d335931303130513036363400000000000000010e4100"
    "0202190033304d335931303130513036363400000000000000b004040603210003"
    "020003000400ea01eb010700cc0d8b1304000a0000000000002420000000000044"
    "fb"
)
FRAME_EVENT_CLEARED_2 = bytes.fromhex(
    "242460000801011c33304d335931303130513036363400000000000000010e3d00"
    "0202190033304d335931303130513036363400000000000000b0040406031d0001"
    "0280ea01eb010700cc0d8b1304000a00000000000024200000000000491c"
)
FRAME_EVENT_CLEARED_3_4 = bytes.fromhex(
    "242462000801011c33304d335931303130513036363400000000000000010e3f00"
    "0202190033304d335931303130513036363400000000000000b0040406031f0002"
    "03800480ea01eb010700cc0d8b1304000a00000000000024200000000000bc76"
)


def _with_data_word(frame: bytes, data_off: int, value: int) -> bytes:
    """Return a copy of ``frame`` with a data-section u16 replaced, CRC fixed."""
    buf = bytearray(frame)
    struct.pack_into("<H", buf, 0x3A + data_off, value)
    struct.pack_into(">H", buf, len(buf) - 2, crc16_buypass(bytes(buf[:-2])))
    return bytes(buf)


class TestRealTimeFrame(unittest.TestCase):
    def test_captured_frame_crc_is_valid(self) -> None:
        self.assertEqual(
            crc16_buypass(FRAME_FW_004_06[:-2]),
            struct.unpack(">H", FRAME_FW_004_06[-2:])[0],
        )

    def test_decodes_frame_from_004_06_firmware(self) -> None:
        parsed = decode_solax_frame(FRAME_FW_004_06)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["wifi_sn"], "30M3Y1010Q0664")
        self.assertEqual(parsed["inverter_sn"], "30M3Y1010Q0664")
        self.assertEqual(parsed["dsp_fw_version"], "004.06")
        self.assertEqual(parsed["rated_power_W"], 1200)
        self.assertEqual(parsed["run_mode"], 1)
        self.assertAlmostEqual(parsed["grid_voltage_V"], 242.7)
        self.assertAlmostEqual(parsed["grid_freq_Hz"], 49.88)
        self.assertEqual(parsed["ac_power_W"], 110)
        self.assertAlmostEqual(parsed["vpv1_V"], 43.7)
        self.assertAlmostEqual(parsed["vpv2_V"], 42.7)
        self.assertAlmostEqual(parsed["e_total_kWh"], 1.5)
        self.assertEqual(parsed["temperature1_C"], 33)
        self.assertEqual(parsed["status_flags"], 3)
        self.assertTrue(parsed["dual_mppt"])
        self.assertEqual(parsed["ppv1_W"], 66)
        self.assertEqual(parsed["ppv2_W"], 70)
        self.assertEqual(parsed["pdc_total_W"], 136)

    def test_decodes_frame_from_005_02_firmware(self) -> None:
        # Same frame with the version word set to what 005.02 units send.
        frame = _with_data_word(FRAME_FW_004_06, 2, 0x0205)
        parsed = decode_solax_frame(frame)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["dsp_fw_version"], "005.02")
        self.assertAlmostEqual(parsed["grid_voltage_V"], 242.7)

    def test_rejects_bad_reserved_word(self) -> None:
        self.assertIsNone(decode_solax_frame(_with_data_word(FRAME_FW_004_06, 5, 0)))

    def test_rejects_bad_crc(self) -> None:
        buf = bytearray(FRAME_FW_004_06)
        buf[-1] ^= 0xFF
        self.assertIsNone(decode_solax_frame(bytes(buf)))

    def test_rejects_boot_frame(self) -> None:
        self.assertIsNone(decode_solax_frame(FRAME_BOOT_FW_STRING))


class TestEventFrame(unittest.TestCase):
    def test_all_captured_frames_are_valid(self) -> None:
        for frame in (
            FRAME_FW_004_06,
            FRAME_BOOT_FW_STRING,
            FRAME_EVENT_RAISED,
            FRAME_EVENT_CLEARED_2,
            FRAME_EVENT_CLEARED_3_4,
        ):
            self.assertTrue(is_valid_solax_frame(frame))
        self.assertFalse(is_valid_solax_frame(FRAME_FW_004_06[:-1]))
        self.assertFalse(is_valid_solax_frame(b"$$"))

    def test_event_frames_are_not_realtime_frames(self) -> None:
        for frame in (
            FRAME_EVENT_RAISED,
            FRAME_EVENT_CLEARED_2,
            FRAME_EVENT_CLEARED_3_4,
        ):
            self.assertIsNone(decode_solax_frame(frame))

    def test_decodes_raised_codes(self) -> None:
        parsed = decode_solax_event_frame(FRAME_EVENT_RAISED)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["run_mode"], 3)
        self.assertEqual(parsed["dsp_fw_version"], "004.06")
        self.assertEqual(parsed["rated_power_W"], 1200)
        self.assertEqual(
            parsed["event_codes"],
            [
                {"code": 2, "cleared": False},
                {"code": 3, "cleared": False},
                {"code": 4, "cleared": False},
            ],
        )

    def test_decodes_cleared_codes(self) -> None:
        parsed = decode_solax_event_frame(FRAME_EVENT_CLEARED_2)
        assert parsed is not None
        self.assertEqual(parsed["event_codes"], [{"code": 2, "cleared": True}])
        parsed = decode_solax_event_frame(FRAME_EVENT_CLEARED_3_4)
        assert parsed is not None
        self.assertEqual(
            parsed["event_codes"],
            [{"code": 3, "cleared": True}, {"code": 4, "cleared": True}],
        )

    def test_event_decoder_rejects_other_frames(self) -> None:
        self.assertIsNone(decode_solax_event_frame(FRAME_FW_004_06))
        self.assertIsNone(decode_solax_event_frame(FRAME_BOOT_FW_STRING))
        buf = bytearray(FRAME_EVENT_RAISED)
        buf[-1] ^= 0xFF
        self.assertIsNone(decode_solax_event_frame(bytes(buf)))


if __name__ == "__main__":
    unittest.main()
