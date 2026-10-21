from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import subtitle_client
import subtitle_daemon


class SplitEtaDashboardTests(unittest.TestCase):
    def setUp(self):
        # Tests must not depend on the user's real learned OCR speed history.
        self._rate_patcher = mock.patch.object(subtitle_daemon, "_load_ocr_rate_hint", return_value=None)
        self._rate_patcher.start()
        self.job_id = "split-eta-test"
        with subtitle_daemon._runtime_lock:
            subtitle_daemon._jobs[self.job_id] = {
                "id": self.job_id,
                "state": "running",
                "options": {"screen_ocr": "off"},
                "asr_work_total": 120.0,
                "overall_audio_done": 30.0,
                "ocr_candidate_work_total": 360.0,
                "ocr_work_total": 0.0,
                "ocr_work_done": 0.0,
                "media_total": 3,
                "ocr_candidate_files": 3,
                "asr_eta_seconds": 45.0,
                "ocr_eta_seconds": None,
                "overall_work_total": 120.0,
                "overall_work_done": 30.0,
                "updated_at": 0.0,
            }

    def tearDown(self):
        with subtitle_daemon._runtime_lock:
            subtitle_daemon._jobs.pop(self.job_id, None)
        self._rate_patcher.stop()

    def test_ocr_toggle_on_recomputes_workload_immediately(self):
        options = subtitle_daemon.set_job_options(
            self.job_id, {"screen_ocr": "speed"}
        )
        self.assertEqual(options["screen_ocr"], "speed")
        with subtitle_daemon._runtime_lock:
            job = dict(subtitle_daemon._jobs[self.job_id])
        self.assertEqual(job["ocr_work_total"], 360.0)
        self.assertEqual(job["ocr_total_files"], 3)
        self.assertEqual(job["overall_work_total"], 480.0)
        self.assertEqual(job["overall_work_done"], 30.0)
        self.assertIsNone(job["ocr_eta_seconds"])
        self.assertIsNone(job["overall_eta_seconds"])

    def test_ocr_toggle_off_removes_ocr_workload(self):
        subtitle_daemon.set_job_options(self.job_id, {"screen_ocr": "speed"})
        with subtitle_daemon._runtime_lock:
            subtitle_daemon._jobs[self.job_id]["ocr_eta_seconds"] = 90.0
            subtitle_daemon._jobs[self.job_id]["ocr_work_done"] = 60.0
        subtitle_daemon.set_job_options(self.job_id, {"screen_ocr": "off"})
        with subtitle_daemon._runtime_lock:
            job = dict(subtitle_daemon._jobs[self.job_id])
        self.assertEqual(job["ocr_work_total"], 0.0)
        self.assertEqual(job["ocr_total_files"], 0)
        self.assertEqual(job["overall_work_total"], 120.0)
        self.assertEqual(job["overall_work_done"], 30.0)
        self.assertAlmostEqual(job["overall_eta_seconds"], 45.0, places=3)

    def test_split_bar_width_follows_estimated_time_ratio(self):
        bar = subtitle_client._split_phase_bar(
            1.0, 0.5, 20.0, 60.0, ocr_enabled=True, width=32
        )
        self.assertTrue(bar.startswith("["))
        self.assertTrue(bar.endswith("]"))
        left, right = bar[1:-1].split("|", 1)
        self.assertEqual(len(left), 8)
        self.assertEqual(len(right), 24)
        self.assertIn("=", left)
        self.assertIn("#", right)

    def test_split_bar_includes_preparation_weight(self):
        bar = subtitle_client._split_phase_bar(
            0.0,
            0.0,
            20.0,
            60.0,
            ocr_enabled=True,
            prep_progress=0.5,
            prep_weight=20.0,
            width=40,
        )
        parts = bar[1:-1].split("|")
        self.assertEqual(len(parts), 3)
        self.assertEqual(sum(len(part) for part in parts), 40)
        self.assertIn("+", parts[0])

    def test_render_contract_contains_unified_counters_and_split_etas(self):
        source = pathlib.Path(subtitle_client.__file__).read_text(encoding="utf-8")
        self.assertIn("[현재 파일]  음성 {asr_counter}/{media_total} · OCR {ocr_counter_text}", source)
        self.assertIn("[현재 단계]  {stage}", source)
        self.assertIn("음성 남은 예상 시간", source)
        self.assertIn("OCR 남은 예상 시간", source)
        self.assertIn("[예상] 과거 평균", source)
        self.assertIn("실측 예상", source)
        self.assertIn("[예상 구성] 준비", source)
        self.assertIn("남은 시간", source)
        self.assertIn("완료 예상 시각", source)
        self.assertIn("[음성 {asr_progress * 100.0:5.1f}%]", source)
        self.assertIn("_split_phase_bar(", source)


if __name__ == "__main__":
    unittest.main()
