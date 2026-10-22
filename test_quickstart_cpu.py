"""Offline tests for standalone, CPU-only subtitle application."""
import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

SRC=pathlib.Path(__file__).resolve().parent/"distribution"/"quickstart_cpu.py"
spec=importlib.util.spec_from_file_location("cpu_quickstart",SRC)
cpu=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)

class QuickstartTests(unittest.TestCase):
    def test_packaged_lgpl_runtime_detects_only_lgpl_shared(self):
        import json
        core = types.ModuleType("av._core")
        core.library_meta = {"libavcodec": {
            "license": "LGPL version 3 or later",
            "configuration": "--enable-shared --disable-static --disable-libx264 --disable-libx265",
        }}
        av_fake = types.ModuleType("av")
        av_fake.__version__ = "17.1.0"
        with tempfile.TemporaryDirectory() as td:
            output = pathlib.Path(td) / "audit.json"
            with mock.patch.dict(sys.modules, {"av": av_fake, "av._core": core}):
                with mock.patch.dict(os.environ, {"SUBTITLE_AUTO_LGPL_REPORT": str(output)}):
                    report = cpu.verify_lgpl_runtime()
            self.assertTrue(report["lgpl_verified"])
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["lgpl_verified"])

    def test_packaged_lgpl_runtime_rejects_gpl_codec(self):
        core = types.ModuleType("av._core")
        core.library_meta = {"libavcodec": {
            "license": "LGPL version 3 or later",
            "configuration": "--enable-shared --enable-libx264",
        }}
        av_fake = types.ModuleType("av")
        av_fake.__version__ = "17.1.0"
        with mock.patch.dict(sys.modules, {"av": av_fake, "av._core": core}):
            with self.assertRaisesRegex(RuntimeError, "LGPL_GATE"):
                cpu.verify_lgpl_runtime()

    def test_packaged_lgpl_runtime_rejects_undocumented_ffmpeg(self):
        core = types.ModuleType("av._core")
        core.library_meta = {"libavcodec": {
            "license": "GPL version 3 or later",
            "configuration": "--enable-shared",
        }}
        av_fake = types.ModuleType("av")
        av_fake.__version__ = "17.1.0"
        with mock.patch.dict(sys.modules, {"av": av_fake, "av._core": core}):
            with self.assertRaisesRegex(RuntimeError, "LGPL_GATE"):
                cpu.verify_lgpl_runtime()

    def test_cpu_explicit(self):
        self.assertEqual(cpu.choose_backend("cpu"), [("cpu", "int8")])

    def test_auto_prefers_cuda(self):
        fake = types.ModuleType("ctranslate2")
        fake.get_supported_compute_types = lambda device: {"float16", "int8"}
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(cpu.choose_backend("auto")[0], ("cuda", "float16"))
            self.assertEqual(cpu.choose_backend("auto")[-1], ("cpu", "int8"))

    def test_auto_falls_back_on_cuda_probe_failure(self):
        fake = types.ModuleType("ctranslate2")
        fake.get_supported_compute_types = lambda device: (_ for _ in ()).throw(RuntimeError("No GPU"))
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(cpu.choose_backend("auto"), [("cpu", "int8")])
            with self.assertRaises(RuntimeError):
                cpu.choose_backend("cuda")

    def test_cli_gpu_failure_does_not_switch_to_cpu(self):
        fake = types.ModuleType("ctranslate2")
        fake.get_supported_compute_types = lambda device: (_ for _ in ()).throw(RuntimeError("No GPU"))
        with mock.patch.dict(sys.modules, {"ctranslate2": fake}):
            with self.assertRaises(RuntimeError):
                cpu.choose_backend("cuda")

    def test_srt_timestamp(self):
        self.assertEqual(cpu.as_srt_time(3723.123),"01:02:03,123")
        self.assertEqual(cpu.as_srt_time(-1),"00:00:00,000")

    def test_srt_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            video=pathlib.Path(td)/"覚-日本語_한글.wav"
            video.write_bytes(b"wav")
            dest=cpu.write_srt(video,[(0,1.5,"안녕하세요")])
            self.assertIn("안녕하세요",dest.read_text(encoding="utf-8"))
            previous=dest.read_bytes()
            with self.assertRaises(FileExistsError):
                cpu.write_srt(video,[(0,1.5,"do not replace")])
            self.assertEqual(previous,dest.read_bytes())

    def test_model_and_file_validation(self):
        with tempfile.TemporaryDirectory() as td:
            src=pathlib.Path(td)/"something.doc"
            src.write_text("data")
            with self.assertRaises(ValueError):cpu.transcribe(src)

    def test_transcribe_with_injected_stub_avoids_internet(self):
        class Segment:
            def __init__(self, start,end,text):
                self.start,self.end,self.text=start,end,text
        class Model:
            def __init__(self,*a,**kw):
                self.kw=kw
            def transcribe(self,*a,**kw):
                return iter([Segment(0.5,1.7,"  This is a test  "),Segment(2,2.6,"좋아요")]),types.SimpleNamespace(language="en")
        lib=types.ModuleType("faster_whisper")
        lib.WhisperModel=Model
        with tempfile.TemporaryDirectory() as td:
            f=pathlib.Path(td)/"sample.wav"
            f.write_bytes(b"wav")
            with mock.patch.dict(sys.modules,{"faster_whisper":lib}):
                with mock.patch.object(cpu,"model_cache",return_value=pathlib.Path(td)/"models"):
                    result=cpu.transcribe(f,model_name="tiny",device="cpu")
            s=result.read_text(encoding="utf-8")
            self.assertIn("00:00:00,500 --> 00:00:01,700",s)
            self.assertIn("This is a test",s)
            self.assertIn("좋아요",s)

if __name__=="__main__":
    unittest.main()
