from __future__ import annotations

import unittest

import screen_ocr


def obs(t, watermark=True, caption=None):
    items = []
    if watermark:
        items.append({
            "text": "IPPA 010054",
            "confidence": 0.95,
            "region": (0.86, 0.04, 0.97, 0.15),
        })
    if caption:
        items.append({
            "text": caption,
            "confidence": 0.98,
            "region": (0.28, 0.72, 0.72, 0.84),
        })
    if not items:
        return {"time": t, "text": "", "confidence": 0.0, "region": None, "items": []}
    return {
        "time": t,
        "text": "\n".join(item["text"] for item in items),
        "confidence": sum(item["confidence"] for item in items) / len(items),
        "region": (
            min(item["region"][0] for item in items),
            min(item["region"][1] for item in items),
            max(item["region"][2] for item in items),
            max(item["region"][3] for item in items),
        ),
        "items": items,
    }


class StaticWatermarkFilterTests(unittest.TestCase):
    def test_persistent_small_region_watermark_is_removed(self):
        observations = []
        for t in range(0, 61):
            caption = None
            if 10 <= t <= 13:
                caption = "실제 화면 자막"
            elif 30 <= t <= 33:
                caption = "두 번째 화면 자막"
            observations.append(obs(float(t), True, caption))

        filtered = screen_ocr._filter_static_watermark_items(observations)
        text = "\n".join(item.get("text", "") for item in filtered)
        self.assertNotIn("IPPA", text)
        self.assertNotIn("010054", text)
        self.assertIn("실제 화면 자막", text)
        self.assertIn("두 번째 화면 자막", text)

        cues = screen_ocr.merge_observations(observations, "speed")
        cue_text = "\n".join(cue[2] for cue in cues)
        self.assertNotIn("IPPA", cue_text)
        self.assertIn("실제 화면 자막", cue_text)

    def test_short_static_title_is_not_removed(self):
        observations = []
        for t in range(0, 61):
            title = None
            if 20 <= t <= 25:
                title = "제1장 시작"
            observations.append(obs(float(t), False, title))

        filtered = screen_ocr._filter_static_watermark_items(observations)
        text = "\n".join(item.get("text", "") for item in filtered)
        self.assertIn("제1장 시작", text)

    def test_changing_text_in_same_subtitle_region_is_not_removed(self):
        observations = []
        for t in range(0, 61):
            observations.append(obs(float(t), False, f"대사 {t}"))
        filtered = screen_ocr._filter_static_watermark_items(observations)
        kept = [item for item in filtered if item.get("text")]
        self.assertGreater(len(kept), 40)

    def test_unstable_top_right_logo_fragments_are_removed(self):
        fragments = ["moodys", "0", "01", "1"]
        observations = []
        for t in range(0, 61):
            items = [{
                "text": fragments[t % len(fragments)],
                "confidence": 0.75,
                "region": (0.86, 0.04, 0.97, 0.15),
            }]
            if 25 <= t <= 28:
                items.append({
                    "text": "실제 화면 자막",
                    "confidence": 0.98,
                    "region": (0.28, 0.72, 0.72, 0.84),
                })
            observations.append({
                "time": float(t),
                "text": "\n".join(item["text"] for item in items),
                "confidence": sum(item["confidence"] for item in items) / len(items),
                "region": (
                    min(item["region"][0] for item in items),
                    min(item["region"][1] for item in items),
                    max(item["region"][2] for item in items),
                    max(item["region"][3] for item in items),
                ),
                "items": items,
            })

        filtered = screen_ocr._filter_static_watermark_items(observations)
        text = "\n".join(item.get("text", "") for item in filtered)
        for fragment in fragments:
            self.assertNotIn(fragment, text)
        self.assertIn("실제 화면 자막", text)

    def test_top_right_logo_region_is_learned_for_pre_ocr_masking(self):
        clusters = []
        masks = []
        fragments = ["moodys", "0", "01", "1"]
        for index, timestamp in enumerate(range(0, 16, 2)):
            screen_ocr._learn_watermark_masks(
                clusters,
                masks,
                float(timestamp),
                {
                    "items": [{
                        "text": fragments[index % len(fragments)],
                        "confidence": 0.75,
                        "region": (0.86, 0.04, 0.97, 0.15),
                    }]
                },
            )
        self.assertEqual(len(masks), 1)
        self.assertTrue(screen_ocr._same_static_region(masks[0], (0.86, 0.04, 0.97, 0.15)))


if __name__ == "__main__":
    unittest.main()
