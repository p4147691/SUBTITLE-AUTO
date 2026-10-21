from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import subtitle_daemon

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "_ocr_existing_srt_real_video"


class DaemonOcrLiveAcceptance(unittest.TestCase):
    def test_one_real_daemon_ocr_job(self):
        with tempfile.TemporaryDirectory(prefix="daemon-ocr-live-") as td:
            target = Path(td)
            shutil.copy2(SOURCE / "existing.mp4", target / "existing.mp4")
            shutil.copy2(SOURCE / "자막" / "외부" / "existing.srt", target / "existing.srt")
            manifest_dir = target / "자막"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            (manifest_dir / ".subtitle-auto.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "items": {
                            "existing.mp4": {
                                "final": "existing.srt",
                                "final_kind": "audio",
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

            job_id = "liveocr_" + uuid.uuid4().hex
            options = subtitle_daemon.normalize_job_options({
                "rename_files": False,
                "lookup_product": False,
                "rename_folder": False,
                "keep_model_loaded": False,
                "screen_ocr": "speed",
            })

            with subtitle_daemon._runtime_lock:
                subtitle_daemon._jobs[job_id] = {
                    "id": job_id,
                    "engine": "qwen",
                    "target": str(target),
                    "state": "queued",
                    "total": 0,
                    "completed": 0,
                    "message": "live acceptance",
                    "options": options,
                    "created_at": 0.0,
                    "updated_at": 0.0,
                    "cancel_requested": False,
                    "ocr_worker_pid": None,
                    "ocr_worker_started_at": None,
                    "ocr_worker_last_exit": None,
                }

            self.assertTrue(subtitle_daemon._job_lock.acquire(blocking=False))
            subtitle_daemon.run_job(job_id, "qwen", target)

            with subtitle_daemon._runtime_lock:
                job = dict(subtitle_daemon._jobs[job_id])

            print("LIVE_OCR_JOB_ID=" + job_id, flush=True)
            print("LIVE_OCR_STATE=" + str(job.get("state")), flush=True)
            print("LIVE_OCR_WORKER_PID=" + str(job.get("ocr_worker_pid")), flush=True)
            print("LIVE_OCR_LAST_EXIT=" + repr(job.get("ocr_worker_last_exit")), flush=True)

            self.assertEqual(job.get("state"), "done", job.get("error") or job.get("message"))
            self.assertIsNone(job.get("ocr_worker_pid"))
            exit_info = job.get("ocr_worker_last_exit")
            self.assertIsInstance(exit_info, dict)
            self.assertEqual(exit_info.get("reason"), "normal")
            self.assertIsNotNone(exit_info.get("worker_pid"))
            self.assertFalse(subtitle_daemon._job_lock.locked())


if __name__ == "__main__":
    unittest.main()
