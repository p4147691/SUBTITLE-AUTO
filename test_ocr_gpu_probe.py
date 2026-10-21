from __future__ import annotations

import json
import subprocess
import unittest

import screen_ocr


class OcrGpuProbeTests(unittest.TestCase):
    def test_gpu_runtime_is_available(self):
        code = (
            "import json,paddle;"
            "compiled=bool(paddle.device.is_compiled_with_cuda());"
            "count=int(paddle.device.cuda.device_count()) if compiled else 0;"
            "device='';"
            "\nif compiled and count>0:\n"
            " paddle.set_device('gpu:0')\n"
            " device=paddle.device.get_device()\n"
            "print(json.dumps({'compiled_with_cuda':compiled,'device_count':count,'device':device,'version':paddle.__version__}))"
        )
        proc = subprocess.run(
            [str(screen_ocr.OCR_PYTHON), "-c", code],
            cwd=str(screen_ocr.ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lines=[x for x in proc.stdout.splitlines() if x.strip()]
        data=json.loads(lines[-1])
        print("OCR_GPU_PROBE="+json.dumps(data,ensure_ascii=False))
        self.assertTrue(data["compiled_with_cuda"], data)
        self.assertGreaterEqual(data["device_count"], 1, data)
        self.assertEqual(data["device"], "gpu:0", data)


if __name__ == "__main__":
    unittest.main()
