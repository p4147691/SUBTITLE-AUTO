from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import screen_ocr


REAL_POPEN = subprocess.Popen


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x00100000, 0, int(pid))  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x00000102  # WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


def _read_pid(path: Path, timeout: float = 5.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            value = int(path.read_text(encoding="utf-8").strip())
            if value > 0:
                return value
        except Exception:
            pass
        time.sleep(0.05)
    raise AssertionError(f"PID file not ready: {path}")


def _worker_script(*, finish: bool) -> str:
    tail = (
        "print('SCREEN_OCR_PROGRESS_JSON=' + json.dumps({'timestamp': 1.0}), flush=True)\n"
        "print('SCREEN_OCR_RESULT_JSON=' + json.dumps({'ok': True}), flush=True)\n"
        if finish
        else "time.sleep(60)\n"
    )
    return (
        "import json, pathlib, subprocess, sys, time\n"
        "print('SCREEN_OCR_PROGRESS_JSON=' + json.dumps({'timestamp': 0.0}), flush=True)\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        + tail
    )


@unittest.skipUnless(os.name == "nt", "Windows process-tree cleanup regression")
class OcrWorkerCleanupTests(unittest.TestCase):
    def test_normal_completion_leaves_no_worker_child(self):
        with tempfile.TemporaryDirectory(prefix="ocr-worker-normal-") as td:
            child_pid_file = Path(td) / "child.pid"
            events = []
            result = screen_ocr._run_worker_process(
                [sys.executable, "-c", _worker_script(finish=True), str(child_pid_file)],
                lifecycle_callback=lambda event, payload: events.append((event, payload)),
                worker_timeout_seconds=10,
                no_progress_timeout_seconds=5,
            )
            child_pid = _read_pid(child_pid_file)
            self.assertTrue(result["ok"])
            self.assertTrue(_wait_dead(child_pid), f"child still alive: {child_pid}")
            self.assertEqual(events[0][0], "started")
            self.assertEqual(events[-1][0], "exited")
            self.assertEqual(events[-1][1]["reason"], "normal")

    def test_no_progress_timeout_kills_worker_tree(self):
        with tempfile.TemporaryDirectory(prefix="ocr-worker-timeout-") as td:
            child_pid_file = Path(td) / "child.pid"
            events = []
            with self.assertRaises(screen_ocr.OcrWorkerTimeout):
                screen_ocr._run_worker_process(
                    [sys.executable, "-c", _worker_script(finish=False), str(child_pid_file)],
                    lifecycle_callback=lambda event, payload: events.append((event, payload)),
                    worker_timeout_seconds=10,
                    no_progress_timeout_seconds=1,
                )
            child_pid = _read_pid(child_pid_file)
            self.assertTrue(_wait_dead(child_pid), f"child still alive: {child_pid}")
            self.assertEqual(events[-1][0], "exited")
            self.assertEqual(events[-1][1]["reason"], "timeout")
            self.assertTrue(events[-1][1]["terminate"] or events[-1][1]["kill"])

    def test_job_object_close_kills_worker_tree_on_daemon_exit(self):
        with tempfile.TemporaryDirectory(prefix="ocr-worker-daemon-exit-") as td:
            child_pid_file = Path(td) / "child.pid"
            proc = REAL_POPEN(
                [sys.executable, "-c", _worker_script(finish=False), str(child_pid_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            guard = screen_ocr._KillOnCloseJob()
            self.assertTrue(guard.assign(proc), "worker could not be assigned to kill-on-close Job Object")
            child_pid = _read_pid(child_pid_file)
            guard.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.fail("worker survived Job Object close")
            self.assertTrue(_wait_dead(child_pid), f"grandchild still alive: {child_pid}")
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

    def test_ffmpeg_error_path_terminates_process_and_closes_pipes(self):
        holder = {}

        def fake_popen(_command, **kwargs):
            proc = REAL_POPEN(
                [
                    sys.executable,
                    "-c",
                    "import sys,time; sys.stdout.buffer.write(b'\\x00'*12); sys.stdout.flush(); time.sleep(60)",
                ],
                stdout=kwargs.get("stdout"),
                stderr=kwargs.get("stderr"),
                creationflags=kwargs.get("creationflags", 0),
            )
            holder["proc"] = proc
            return proc

        def raise_progress(_timestamp):
            raise RuntimeError("forced-progress-error")

        with mock.patch.object(screen_ocr, "_video_geometry", return_value=(2, 2)), \
             mock.patch.object(screen_ocr, "_scaled_geometry", return_value=(2, 2)), \
             mock.patch.object(screen_ocr, "_load_ocr", return_value=object()), \
             mock.patch.object(
                 screen_ocr,
                 "_ocr_frame",
                 return_value={"text": "x", "confidence": 1.0, "region": (0.0, 0.0, 1.0, 1.0)},
             ), \
             mock.patch.object(screen_ocr.shutil, "which", return_value="ffmpeg"), \
             mock.patch.object(screen_ocr.subprocess, "Popen", side_effect=fake_popen):
            with self.assertRaisesRegex(RuntimeError, "forced-progress-error"):
                screen_ocr.extract_screen_cues(Path("fake.mp4"), "speed", progress_callback=raise_progress)

        proc = holder["proc"]
        self.assertIsNotNone(proc.poll(), "ffmpeg test process still alive")
        self.assertTrue(proc.stdout is None or proc.stdout.closed)
        self.assertTrue(proc.stderr is None or proc.stderr.closed)


if __name__ == "__main__":
    unittest.main()
