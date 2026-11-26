import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common import write_csv
from src.evaluation.gallery import clustered_rate, evaluate


class GalleryTests(unittest.TestCase):
    def test_clustered_ci_resamples_identities_and_weights_images(self):
        result = clustered_rate([0, 0, 0, 1], ["a", "a", "a", "b"], 500, 13)
        self.assertEqual(result["rate"], 0.25)
        self.assertEqual((result["ci95_low"], result["ci95_high"]), (0.0, 1.0))
        self.assertIsNone(clustered_rate([0, 0], ["a", "b"], 50, 1)["ci95_high"])

    def test_nested_gallery_preserves_cohort_and_separates_error_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = np.random.default_rng(19).permutation(["alice", "bob"]).tolist()
            rows, vectors = [], []
            def add(person, partition, role, xy):
                image = f"photo{len(rows)}"
                rows.append(dict(identity_id=person, image_id=image, partition=partition,
                                 role=role, relative_path=image + ".jpg"))
                vectors.append(xy + [0.] * 510)
            add(names[0], "G", "enrollment", [1., 0.])
            add(names[1], "G", "enrollment", [0., 1.])
            add(names[0], "G", "held_out", [1., 0.])
            add(names[0], "G", "held_out", [0., 1.])
            add(names[0], "G", "held_out", [-1., 0.])
            add(names[1], "G", "held_out", [0., 1.])
            add("stranger", "D", "development", [1., 0.])
            write_csv(root / "index.csv", rows)
            np.savez(root / "vectors.npz", embeddings=np.array(vectors, np.float32),
                     image_ids=[r["image_id"] for r in rows])
            (root / "cal.json").write_text(json.dumps({
                "settings": {"gallery_size": 2}, "operating_points": {"0.01": {"threshold": 1.0}}}))
            config = {"paths": {"index": root / "index.csv", "embeddings": root / "vectors.npz",
                                "image_root": root, "results": root, "calibration": root / "cal.json"},
                      "evaluation": {"gallery_sizes": [1, 2], "seed": 19,
                                     "bootstrap_repetitions": 20, "block_size": 2}}
            out = evaluate(config)
            def read(name):
                with (out / name).open(newline="") as f:
                    return list(csv.DictReader(f))
            self.assertEqual([r["known_probes"] for r in read("gallery_sizes.csv")], ["3", "3"])
            largest = {r["event"]: (int(r["events"]), int(r["images"]))
                       for r in read("thresholds.csv") if r["gallery_size"] == "2"}
            self.assertEqual(largest, {"false_reject": (1, 3), "wrong_identity_accepted": (1, 3),
                                       "false_accept": (1, 1)})
            self.assertEqual(len(read("errors.csv")), 3)
            self.assertTrue(all(r["brightness_band"] == "unavailable" for r in read("brightness.csv")))
