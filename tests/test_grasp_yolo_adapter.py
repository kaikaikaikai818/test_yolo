import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import torch
from bsp.camera_bsp.yolo_tool_detect import YoloEngine, canonical_name
from bsp.camera_bsp.sam_tool_detect import SamToolDetector

class AdapterTests(unittest.TestCase):
    def engine(self, shape=(48, 64)):
        mask = torch.zeros((1, *shape))
        mask[0, 10:35, 10:40] = 1
        mask[0, 18:22, 18:22] = 0
        box = SimpleNamespace(xyxy=torch.tensor([[10,10,40,35]]),
                              conf=torch.tensor(.9), cls=torch.tensor(3))
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=[box], masks=SimpleNamespace(data=mask))]
        engine = YoloEngine.__new__(YoloEngine)
        engine.model, engine.class_ids, engine.device = model, [3], "cpu"
        return engine

    def test_mask_hole_preserved_and_target_filtered(self):
        engine = self.engine()
        out = engine.infer(np.zeros((48,64,3),np.uint8))
        self.assertFalse(out.segmentations[0].mask[19,19])
        self.assertTrue(out.segmentations[0].mask[15,15])
        self.assertEqual(engine.model.predict.call_args.kwargs["classes"], [3])
        engine.class_ids = [2]
        self.assertEqual(engine.infer(np.zeros((48,64,3),np.uint8)).segmentations, [])

    def test_wrong_mask_size_rejected(self):
        with self.assertRaises(ValueError):
            self.engine((40,64)).infer(np.zeros((48,64,3),np.uint8))

    def test_roi_and_depth_units(self):
        detector = SamToolDetector.__new__(SamToolDetector)
        detector.engine = self.engine()
        detector.prompt = "screwdriver"
        detector.min_depth_m, detector.max_depth_m = .15, 3
        depth = np.full((100,120), 500, np.float32)
        result = detector.detect_roi(np.zeros((100,120,3),np.uint8), depth, .001, (20,30,84,78))
        self.assertEqual(result["mask"].shape, (100,120))
        self.assertEqual(result["box"], (30,40,60,65))
        self.assertAlmostEqual(result["z_mm"],500)
        self.assertFalse(result["mask"][:30].any())
        depth[:] = 0
        self.assertIsNone(detector.detect_roi(np.zeros((100,120,3),np.uint8), depth,.001,(20,30,84,78))["z_mm"])

    def test_aliases(self):
        self.assertEqual(canonical_name("an adjustable wrench"),"wrench")
        self.assertEqual(canonical_name("Screwdriver"),"screwdriver")
        self.assertEqual(canonical_name("a tape dispenser"),"tape cutter")
