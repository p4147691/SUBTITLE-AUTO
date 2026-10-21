"""Regression contract for Korean Windows with the UTF-8 system-locale toggle ON or OFF."""
from __future__ import annotations
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LAUNCHERS=("run_qwen.cmd","run_whisper.cmd","run_sensevoice.cmd",
           "run_ocr.cmd","run_gpu_memory_viewer.cmd")
SOURCES=("subtitle_client.py","engine_qwen.py","ocr_client.py","gpu_memory_viewer.py")

class WindowsLocaleContractTests(unittest.TestCase):
    def test_launcher_contract(self):
        for filename in LAUNCHERS:
            with self.subTest(file=filename):
                text=(ROOT/filename).read_text(encoding="utf-8").casefold()
                self.assertIn("chcp 65001",text)
                self.assertIn("set pythonutf8=1",text)
                self.assertIn("set pythonioencoding=utf-8",text)
                self.assertIn("%~dp0",text)
                self.assertNotIn("d:\\ai_tools\\subtitle-auto",text)

    def test_process_text_decoders_are_explicit(self):
        for filename in SOURCES:
            tree=ast.parse((ROOT/filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node,ast.Call):continue
                call=node.func
                if not (isinstance(call,ast.Attribute) and call.attr=="run"
                    and isinstance(call.value,ast.Name) and call.value.id=="subprocess"):continue
                kw={item.arg:item.value for item in node.keywords}
                if isinstance(kw.get("text"),ast.Constant) and kw["text"].value is True:
                    with self.subTest(file=filename,line=node.lineno):
                        self.assertEqual(kw.get("encoding").value,"utf-8")
                        self.assertEqual(kw.get("errors").value,"replace")

    def test_real_unicode_media_under_both_modes(self):
        import engine_qwen
        import common
        exe=shutil.which("ffmpeg")
        if not exe or not shutil.which("ffprobe"):self.skipTest("ffmpeg/ffprobe unavailable")
        child=r"""
import json,sys,tempfile,subprocess
from pathlib import Path
import engine_qwen,common
with tempfile.TemporaryDirectory(prefix='locale-corpus-') as td:
 p=Path(td);a=p/('覚_日本語_한글.wav');b=p/'chunk.wav'
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i','sine=frequency=501:sample_rate=16000:duration=0.35','-ac','1','-c:a','pcm_s16le',str(a)]
 res=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15)
 assert res.returncode==0,res.stderr[:100]
 d=engine_qwen.ffprobe_duration(a)
 engine_qwen.extract_chunk(a,b,0.0,0.25)
 common.write_srt_atomic(a.with_suffix('.srt'),[(0,0.24,'日本語 한글')])
 assert 0.30<=d<=0.40 and b.exists()
 assert '日本語' in a.with_suffix('.srt').read_text(encoding='utf-8')
 common.configure_unicode_stdio()
 print(json.dumps({'pass':True,'utf8_mode':sys.flags.utf8_mode,'stdout':sys.stdout.encoding}))
"""
        # The child code must contain real Unicode and remain safe in a legacy codepage stream.
        for name,flag,codec in (("cp949_off","0","cp949"),("utf8_on","1","utf-8")):
            with self.subTest(mode=name):
                env=dict(os.environ,PYTHONUTF8=flag,PYTHONIOENCODING=codec)
                run=subprocess.run([sys.executable,"-c",child],cwd=ROOT,env=env,
                    capture_output=True,timeout=40)
                self.assertEqual(run.returncode,0,run.stderr.decode("utf-8","replace")[-500:])
                payload=json.loads(run.stdout.decode(codec))
                self.assertTrue(payload["pass"])
                self.assertEqual(payload["utf8_mode"],int(flag))

    def test_cp949_console_does_not_crash(self):
        script="import common;common.configure_unicode_stdio();print('覚 日本語 한글')"
        for codec,flag in (("cp949","0"),("utf-8","1")):
            with self.subTest(codec=codec):
                p=subprocess.run([sys.executable,"-c",script],cwd=ROOT,
                   env=dict(os.environ,PYTHONUTF8=flag,PYTHONIOENCODING=codec),
                   capture_output=True,timeout=15)
                self.assertEqual(p.returncode,0,p.stderr.decode("utf-8","replace"))
                self.assertTrue(p.stdout)

if __name__=="__main__":
    unittest.main()
