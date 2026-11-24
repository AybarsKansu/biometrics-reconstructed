"""Local standard 6,000-pair / 10-fold LFW check; never drops failed pairs."""
import csv
from pathlib import Path

import numpy as np

from common import load_config, load_data, write_json


def image_id(name, number):
    return f"{name}_{int(number):04d}"


def read_pairs(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    pairs = []
    if path.suffix.lower() == ".csv":
        # Kaggle's ragged CSV: same rows name,n1,n2; different rows name1,n1,name2,n2.
        reader = csv.reader(lines)
        header = next(reader)
        if header[:3] != ["name", "imagenum1", "imagenum2"]:
            raise ValueError("Unsupported pair CSV header; use the official pairs.txt or the Kaggle pairs.csv.")
        records = ([cell for cell in row if cell.strip()] for row in reader)
    else:
        if lines[0].split() != ["10", "300"]:
            raise ValueError("Expected official pairs.txt header: 10 300.")
        records = (line.split() for line in lines[1:] if line.strip())
    for cells in records:
        if len(cells) == 3:
            name, first, second = cells
            pairs.append((image_id(name, first), image_id(name, second), True))
        elif len(cells) == 4:
            first_name, first, second_name, second = cells
            pairs.append((image_id(first_name, first), image_id(second_name, second), False))
        else:
            raise ValueError("Malformed LFW pair row.")
    if len(pairs) != 6000:
        raise ValueError(f"Expected 6000 standard pairs; found {len(pairs)}.")
    # Reject concatenated/sorted mirrors that destroy the original fold ordering.
    for fold in range(10):
        labels = [entry[2] for entry in pairs[fold * 600:(fold + 1) * 600]]
        if labels != [True] * 300 + [False] * 300:
            raise ValueError("Pair order is not the standard ten folds (300 same then 300 different). Supply official pairs.txt.")
    return pairs


def cross_validate(scores, labels, folds=10, pairs_per_fold=600, distance_step=0.01):
    # On unit vectors squared Euclidean distance is 2 - 2*cosine.
    distances = 2 - 2 * np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    thresholds = np.arange(0, 4 + distance_step / 2, distance_step)
    results = []
    for fold in range(folds):
        test = np.arange(fold * pairs_per_fold, (fold + 1) * pairs_per_fold)
        train = np.ones(len(scores), dtype=bool)
        train[test] = False
        accuracies = [np.mean((distances[train] < threshold) == labels[train]) for threshold in thresholds]
        best = float(thresholds[int(np.argmax(accuracies))])
        accuracy = float(np.mean((distances[test] < best) == labels[test]))
        results.append({"fold": fold, "squared_distance_threshold": best, "accuracy": accuracy})
    return results


def check(config):
    _, vectors, rows = load_data(config)
    pairs = read_pairs(config["paths"]["lfw_pairs"])
    positions = {row["image_id"]: i for i, row in enumerate(rows)}
    missing = sorted({image for a, b, _ in pairs for image in (a, b) if image not in positions})
    if missing:
        result = {"status": "blocked_missing_embeddings", "missing_image_ids": missing,
                  "missing_pair_count": sum(a not in positions or b not in positions for a, b, _ in pairs),
                  "accuracy": None, "message": "No failed pair was dropped; a full standard result is unavailable."}
    else:
        scores = [float(np.clip(np.dot(vectors[positions[a]], vectors[positions[b]]), -1, 1)) for a, b, _ in pairs]
        folds = cross_validate(scores, [same for _, _, same in pairs], **config["lfw"])
        result = {"status": "complete", "pairs": len(pairs), "folds": folds,
                  "accuracy": float(np.mean([fold["accuracy"] for fold in folds])),
                  "fold_standard_deviation": float(np.std([fold["accuracy"] for fold in folds])),
                  "protocol": "Original ten folds; squared-distance threshold selected on nine folds; no flip augmentation or mean subtraction"}
    write_json(config["paths"]["lfw_result"], result)
    print(result)


if __name__ == "__main__":
    check(load_config())
