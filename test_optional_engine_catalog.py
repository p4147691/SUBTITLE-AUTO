"""Optional models are cataloged, NOT silently downloaded or claimed runnable."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("alt_s_engine_catalog", ROOT/"distribution"/"engine_catalog.py")
import sys
catalog = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = catalog
spec.loader.exec_module(catalog)


class OptionalEngineTests(unittest.TestCase):
    def test_sensevoice_model_license_is_not_mit(self):
        plan = catalog.installation_plan("sensevoice")
        self.assertEqual(plan["model"], "FunAudioLLM/SenseVoiceSmall")
        self.assertIn("FunASR", plan["license"])
        self.assertFalse(plan["runtime_available"])

    def test_qwen_asr_large_model_requires_separate_runtime(self):
        plan = catalog.installation_plan("qwen")
        self.assertTrue(plan["approx_download_gb"] >= 4.0)
        self.assertEqual(plan["license"], "Apache-2.0")
        self.assertFalse(plan["runtime_available"])

    def test_no_unverified_model_only_install_as_finished_engine(self):
        for key in ("sensevoice","qwen"):
            with self.subTest(engine=key), self.assertRaisesRegex(RuntimeError,"ENGINE_RUNTIME_NOT_PACKAGED"):
                catalog.assert_can_offer_working_download(key)


if __name__=="__main__":
    unittest.main()
