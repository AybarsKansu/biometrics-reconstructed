"""Small file, configuration, and vector helpers shared by the scripts."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path="config.json"):
    path = Path(path).resolve()
    config = read_json(path)
    for name, value in config["paths"].items():
        if value is not None:
            config["paths"][name] = str((path.parent / value).resolve())
    if config["model"]["name"] != "buffalo_l" or config["model"]["dimension"] != 512:
        raise ValueError("This project uses buffalo_l and 512 dimensions only.")
    return config


def read_index(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("Expected one finite 512-dimensional vector.")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm <= 0:
        raise ValueError("Cannot normalize a zero vector.")
    return (vector / norm).astype(np.float32)


def check_embeddings(vectors, image_ids):
    if vectors.dtype != np.float32 or vectors.shape != (len(image_ids), 512):
        raise ValueError("Embedding store must contain float32 N x 512 vectors.")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate embedding image IDs.")
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite embedding.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError("Embeddings must be L2 normalized.")


def load_data(config):
    rows = read_index(config["paths"]["index"])
    with np.load(config["paths"]["embeddings"], allow_pickle=False) as store:
        vectors = store["embeddings"]
        image_ids = store["image_ids"].tolist()
    by_id = {row["image_id"]: row for row in rows}
    aligned_rows = [by_id[image_id] for image_id in image_ids]
    return rows, vectors, aligned_rows


def select(vectors, rows, partition=None, role=None):
    mask = np.array([(partition is None or row["partition"] == partition) and
                     (role is None or row["role"] == role) for row in rows], dtype=bool)
    return vectors[mask], [row for row, chosen in zip(rows, mask) if chosen]


def validate_roles(rows, split, final=False):
    groups = {}
    for row in rows:
        identity = row["identity_id"]
        if identity in groups and groups[identity] != row["partition"]:
            raise ValueError("An identity crosses partitions.")
        groups[identity] = row["partition"]
    counts = Counter(groups.values())
    if any(counts[group] != split[group] for group in ("G", "P", "H")):
        raise ValueError(f"Wrong partition identity counts: {dict(counts)}")
    if final:
        for identity, group in groups.items():
            if group == "G":
                usable = [r for r in rows if r["identity_id"] == identity and
                          r["embedding_status"] == "ok"]
                if sum(r["role"] == "enrollment" for r in usable) != 1:
                    raise ValueError("G identity needs exactly one usable enrollment.")
                if not any(r["role"] == "held_out" for r in usable):
                    raise ValueError("G identity needs held-out data.")
    return dict(counts)


