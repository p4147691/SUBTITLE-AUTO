from __future__ import annotations

import unittest

import screen_ocr


def make_obs(t, text, conf=0.75, region=(0.45, 0.60, 0.50, 0.64)):
    return {
        "time": float(t),
        "text": text,
        "confidence": conf,
        "region": region,
        "items": [{"text": text, "confidence": conf, "region": region}],
    }


class ShortAlnumNoiseTests(unittest.TestCase):
    def test_transient_mv_is_removed(self):
        observations = [
            make_obs(0.00, "mV", 0.78),
            make_obs(0.25, "mV", 0.80),
            {"time": 0.50, "text": "", "confidence": 0.0, "region": None, "items": []},
        ]
        filtered = screen_ocr._filter_short_alnum_noise(observations)
        self.assertFalse(any("mV" in obs.get("text", "") for obs in filtered))

    def test_single_digits_are_removed_when_brief(self):
        observations = [
            make_obs(0.00, "5", 0.84),
            make_obs(0.25, "5", 0.82),
        ]
        filtered = screen_ocr._filter_short_alnum_noise(observations)
        self.assertFalse(any(obs.get("text") for obs in filtered))

    def test_stable_high_conf_ok_is_kept(self):
        observations = [
            make_obs(0.00, "OK", 0.97),
            make_obs(0.50, "OK", 0.96),
            make_obs(1.00, "OK", 0.98),
            make_obs(1.50, "OK", 0.97),
        ]
        filtered = screen_ocr._filter_short_alnum_noise(observations)
        self.assertTrue(any(obs.get("text") == "OK" for obs in filtered))

    def test_cjk_short_text_is_not_touched(self):
        observations = [make_obs(0.00, "응", 0.75)]
        filtered = screen_ocr._filter_short_alnum_noise(observations)
        self.assertEqual(filtered[0]["text"], "응")


if __name__ == "__main__":
    unittest.main()
