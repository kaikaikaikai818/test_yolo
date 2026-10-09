import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from grasp_preflight import preflight

class PreflightTests(unittest.TestCase):
    def test_missing_configuration_rejects_motion_without_connection(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("importlib.util.find_spec", return_value=object()):
                report=preflight(Path(root),Path(root)/"best.pt","grasp")
            self.assertFalse(report["hardware_connected"])
            self.assertTrue(any("camera_alignment.json" in e for e in report["errors"]))
            self.assertTrue(any("grasp_surface_calibration.json" in e for e in report["errors"]))

    def test_vision_reports_optional_calibration_as_warning(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("importlib.util.find_spec", return_value=object()):
                report=preflight(Path(root),Path(root)/"best.pt","vision")
            self.assertEqual(len(report["warnings"]),2)
            self.assertFalse(any("camera_alignment.json" in e for e in report["errors"]))
