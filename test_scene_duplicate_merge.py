from __future__ import annotations

import unittest

import screen_ocr


REGION = (0.45, 0.55, 0.62, 0.70)


def obs(t, text, conf=0.95):
    if not text:
        return {"time": float(t), "text": "", "confidence": 0.0, "region": None, "items": []}
    return {
        "time": float(t),
        "text": text,
        "confidence": conf,
        "region": REGION,
        "items": [{"text": text, "confidence": conf, "region": REGION}],
    }


class SceneDuplicateMergeTests(unittest.TestCase):
    def test_same_text_survives_brief_scene_change_gap_as_one_cue(self):
        observations = [
            obs(0.0, "mOOD\n분위기"),
            obs(0.5, "mOOD\n분위기"),
            obs(1.0, "mOOD\n분위기"),
            obs(1.5, ""),
            obs(2.0, ""),
            obs(2.5, "mOOD\n분위기"),
            obs(3.0, "mOOD\n분위기"),
        ]
        cues = screen_ocr.merge_observations(observations, "speed")
        self.assertEqual(len(cues), 1)
        self.assertIn("분위기", cues[0][2])

    def test_small_non_numeric_ocr_wobble_stays_one_cue(self):
        observations = [
            obs(0.0, "mOOD\n분위기", 0.93),
            obs(0.5, "mOOD\n분위기", 0.94),
            obs(1.0, "mOOO\n분위기", 0.80),
            obs(1.5, "mOOD\n분위기", 0.96),
        ]
        cues = screen_ocr.merge_observations(observations, "speed")
        self.assertEqual(len(cues), 1)
        self.assertIn("분위기", cues[0][2])

    def test_distinct_text_still_splits(self):
        observations = [
            obs(0.0, "첫 번째 문구"),
            obs(0.5, "첫 번째 문구"),
            obs(1.0, "완전히 다른 문구"),
            obs(1.5, "완전히 다른 문구"),
        ]
        cues = screen_ocr.merge_observations(observations, "speed")
        self.assertEqual(len(cues), 2)


if __name__ == "__main__":
    unittest.main()
