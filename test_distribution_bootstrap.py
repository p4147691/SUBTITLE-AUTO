import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("subtitle_dist_bootstrap", ROOT/"distribution"/"bootstrap.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class DistributionTests(unittest.TestCase):
    def test_check_does_not_create_config(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(module,"ROOT",Path(td)):
            values=module.diagnostic()
            self.assertIn("python_compatible",values)
            self.assertFalse((Path(td)/"config.json").exists())
    def test_config_missing_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/"distribution").mkdir()
            (root/"distribution"/"config.example.json").write_text(
                json.dumps({"qwen_model":"","qwen_python":""}),encoding="utf-8")
            with mock.patch.object(module,"ROOT",root):
                self.assertEqual(module.create_local_config(),"CREATED")
                self.assertEqual(module.create_local_config(),"PRESERVED_EXISTING")
                self.assertTrue(str(root) in json.loads((root/"config.json").read_text(encoding="utf-8"))["qwen_model"])
    def test_unverified_models_are_blocked(self):
        with self.assertRaisesRegex(ValueError,"LICENSE"):
            module.download_models("sensevoice",True,False)
        with self.assertRaisesRegex(RuntimeError,"NOT_VERIFIED"):
            module.download_models("sensevoice",True,True)

if __name__=="__main__": unittest.main()
