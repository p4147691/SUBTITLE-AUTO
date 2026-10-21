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
                    result=cpu.transcribe(f,model_name="tiny")
            s=result.read_text(encoding="utf-8")
            self.assertIn("00:00:00,500 --> 00:00:01,700",s)
            self.assertIn("This is a test",s)
            self.assertIn("좋아요",s)

if __name__=="__main__":
    unittest.main()
