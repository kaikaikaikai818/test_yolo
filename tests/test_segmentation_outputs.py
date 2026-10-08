"""Check that saved camera evidence retains full-frame coordinates and raw masks."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from segmentation_outputs import save_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.context = {"camera": {"role": "d455"}, "model": {"sha256": "test-model"}}

    def test_crop_masks_and_boxes_use_full_frame_coordinates(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        crop = frame[20:60, 30:90]
        masks = np.zeros((2, 40, 60), dtype=np.float32)
        masks[0, 3:18, 4:25] = 1
        masks[0, 7:12, 9:16] = 0  # Keep the mask's hole.
        masks[0, 27:33, 45:51] = 1  # Keep a separate component without a bridge.
        masks[1, 20:35, 8:30] = 1
        boxes = torch.tensor([[4, 3, 51, 33, 0.8, 0], [8, 20, 30, 35, 0.7, 1]])
        result = Results(crop, path="crop.jpg", names={0: "wrench", 1: "pliers"},
                         boxes=boxes, masks=torch.from_numpy(masks))
        display = np.full_like(frame, 100)
        report_path = save_snapshot(self.output, frame, display, result,
                                    (30, 20, 90, 60), self.context)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        root = report_path.parent.parent

        self.assertEqual(report["roi_xyxy"], [30, 20, 90, 60])
        self.assertEqual(report["instances"][0]["box_xyxy"], [34, 23, 81, 53])
        self.assertEqual(report["instances"][1]["box_xyxy"], [38, 40, 60, 55])
        self.assertEqual(report["coordinate_system"], "full_color_frame_pixels")
        self.assertEqual(report["model"]["sha256"], "test-model")
        for index, instance in enumerate(report["instances"]):
            saved_mask = cv2.imread(str(root / instance["mask_file"]), cv2.IMREAD_GRAYSCALE)
            expected = np.zeros((80, 120), dtype=np.uint8)
            expected[20:60, 30:90] = (masks[index] * 255).astype(np.uint8)
            np.testing.assert_array_equal(saved_mask, expected)
            self.assertEqual(instance["mask_area_pixels"], int(np.count_nonzero(expected)))
        np.testing.assert_array_equal(cv2.imread(str(root / report["raw_image"])), frame)
        np.testing.assert_array_equal(cv2.imread(str(root / report["prediction_image"])), display)

    def test_missed_detections_can_be_saved_without_overwriting(self):
        frame = np.zeros((60, 90, 3), dtype=np.uint8)
        result = Results(frame, path="empty.jpg", names={0: "wrench"},
                         boxes=torch.empty((0, 6)), masks=None)
        reports = [save_snapshot(self.output, frame, frame.copy(), result,
                                 (0, 0, 90, 60), self.context) for _ in range(2)]
        self.assertNotEqual(reports[0], reports[1])
        for report_path in reports:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["instances"], [])
            self.assertTrue((report_path.parent.parent / report["raw_image"]).is_file())
            self.assertTrue((report_path.parent.parent / report["prediction_image"]).is_file())
        self.assertEqual(len(list((self.output / "d455" / "images").glob("*.jpg"))), 2)
        self.assertEqual(list((self.output / "d455" / "masks").glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
