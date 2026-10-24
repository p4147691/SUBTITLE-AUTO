from __future__ import annotations

import ast
import unittest
from pathlib import Path

from ocr_acceleration_policy import allow_tensorrt_fp16, disabled_reason


class OCRAccelerationPolicyTests(unittest.TestCase):
    def test_blackwell_bypasses_tensorrt_to_use_fp32(self):
        for major in (12, 13):
            with self.subTest(major=major):
                self.assertFalse(allow_tensorrt_fp16("fp16", "gpu:0", False, (major, 0)))
                self.assertIn("BLACKWELL", disabled_reason("fp16", "gpu:0", False, (major, 0)))

    def test_older_supported_gpu_keeps_fp16(self):
        for major in (7, 8, 9, 10, 11):
            with self.subTest(major=major):
                self.assertTrue(allow_tensorrt_fp16("fp16", "gpu:0", False, (major, 0)))
                self.assertIsNone(disabled_reason("fp16", "gpu:0", False, (major, 0)))

    def test_unknown_capability_uses_safe_fp32(self):
        for cap in (None, (), ("n/a", "n/a")):
            with self.subTest(cap=cap):
                self.assertFalse(allow_tensorrt_fp16("fp16", "gpu:0", False, cap))
                self.assertTrue(disabled_reason("fp16", "gpu:0", False, cap))

    def test_explicit_fp32_and_cpu_never_request_tensorrt(self):
        for args in (
            ("fp32", "gpu:0", False, (12, 0)),
            ("fp16", "cpu", False, None),
            ("fp16", "gpu:0", True, (9, 0)),
        ):
            with self.subTest(args=args):
                self.assertFalse(allow_tensorrt_fp16(*args))
                self.assertIsNone(disabled_reason(*args))

    def test_balanced_profile_wires_safe_gate_and_keeps_models(self):
        source = Path(__file__).resolve().parent.joinpath("screen_ocr.py").read_text("utf-8")
        ast.parse(source)
        self.assertIn("allow_tensorrt_fp16(", source)
        self.assertIn('"PP-OCRv6_medium_det"', source)
        self.assertIn('"PP-OCRv6_medium_rec"', source)
        self.assertIn('"sample_interval": 0.125', source)


if __name__ == "__main__":
    unittest.main()
