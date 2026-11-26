"""Explicit, potentially costly H calibration. No downloads or automatic runs."""
from collections import Counter, defaultdict

import numpy as np

from src.common import (load_config, load_data, select,
                    write_csv, write_json)


def threshold_from_tail(tail, total, target):
    """Conservative empirical cutoff for acceptance defined with >=, including ties.

    Keep floor(target*n)+1 largest observations. Moving one float32 step above
    the smallest retained observation allows at most floor(target*n) errors.
    """
    allowed = int(np.floor(target * total))
    if total < 1 or len(tail) < allowed + 1:
        raise ValueError("Insufficient scores for threshold calibration.")
    boundary = np.partition(tail, len(tail) - allowed - 1)[len(tail) - allowed - 1]
    threshold = float(np.nextafter(np.float32(boundary), np.float32(np.inf)))
    return {"threshold": threshold, "target_fmr": target,
            "achieved_fmr": float(np.count_nonzero(tail >= threshold) / total),
            "trials": total, "allowed_errors": allowed,
            "tail_resolution": 1 / total}


def system_scores(vectors, identities, settings):
    by_identity = defaultdict(list)
    for row, identity in enumerate(identities):
        by_identity[identity].append(row)
    names = sorted(by_identity)
    size = settings["gallery_size"]
    if len(names) <= size:
        raise ValueError("H needs more usable identities than a surrogate gallery.")
    rng = np.random.default_rng(settings["seed"])
    impostor, genuine = [], []
    block = settings["block_size"]
    for number in range(settings["surrogate_galleries"]):
        chosen = set(rng.choice(names, size, replace=False).tolist())
        enrollment = [int(rng.choice(by_identity[name])) for name in sorted(chosen)]
        enrolled_rows = set(enrollment)
        gallery = vectors[enrollment]
        for offset in range(0, len(vectors), block):
            positions = list(range(offset, min(offset + block, len(vectors))))
            scores = np.clip(vectors[positions] @ gallery.T, -1, 1).max(axis=1)
            for position, score in zip(positions, scores):
                if identities[position] not in chosen:
                    impostor.append(score)
                elif position not in enrolled_rows:
                    genuine.append(score)
        print(f"H surrogate gallery {number + 1}/{settings['surrogate_galleries']}", flush=True)
    return np.asarray(impostor, np.float32), np.asarray(genuine, np.float32)


def pairwise_scores(vectors, identities, settings, edges):
    counts = Counter(identities)
    total = (len(vectors) ** 2 - sum(count ** 2 for count in counts.values())) // 2
    if total < 1:
        raise ValueError("H has no different-identity pairs.")
    keep = int(np.floor(settings["recovery_fmr"] * total)) + 1
    tail = np.empty(0, dtype=np.float32)
    impostor_hist = np.zeros(len(edges) - 1, dtype=np.int64)
    genuine_hist = np.zeros_like(impostor_hist)
    block = settings["block_size"]
    ids = np.asarray(identities)
    observed = 0
    for start in range(0, len(vectors), block):
        end = min(start + block, len(vectors))
        for other in range(start, len(vectors), block):
            other_end = min(other + block, len(vectors))
            scores = np.clip(vectors[start:end] @ vectors[other:other_end].T, -1, 1)
            upper = np.arange(start, end)[:, None] < np.arange(other, other_end)[None, :]
            different = ids[start:end, None] != ids[None, other:other_end]
            impostor = scores[upper & different]
            genuine = scores[upper & ~different]
            observed += len(impostor)
            impostor_hist += np.histogram(impostor, bins=edges)[0]
            genuine_hist += np.histogram(genuine, bins=edges)[0]
            tail = np.concatenate([tail, impostor])
            if len(tail) > keep:
                tail = np.partition(tail, len(tail) - keep)[-keep:]
    if observed != total:
        raise ValueError("Pairwise block accounting mismatch.")
    result = threshold_from_tail(tail, total, settings["recovery_fmr"])
    return result, impostor_hist, genuine_hist


def calibrate(config):
    _, vectors, rows = load_data(config)
    h, h_rows = select(vectors, rows, partition="H")
    settings = config["calibration"]
    if settings["surrogate_galleries"] < 1 or settings["block_size"] < 1:
        raise ValueError("Gallery count and block size must be positive.")
    identities = [row["identity_id"] for row in h_rows]
    impostor, genuine = system_scores(h, identities, settings)
    if len(impostor) == 0:
        raise ValueError("No H impostor probes.")
    lo = float(np.percentile(impostor, settings["lo_percentile"]))
    operating_points = {}
    for target in settings["system_fmrs"]:
        result = threshold_from_tail(impostor, len(impostor), target)
        result["lo"] = lo
        if not -1 <= lo < result["threshold"] <= 1:
            raise ValueError("Degenerate H scores cannot define the requested quantizer.")
        operating_points[str(target)] = result
    edges = np.linspace(-1, 1, settings["histogram_bins"] + 1)
    recovery, pair_impostor, pair_genuine = pairwise_scores(h, identities, settings, edges)
    histograms = {"system_impostor": np.histogram(impostor, edges)[0],
                  "system_genuine": np.histogram(genuine, edges)[0],
                  "pair_impostor": pair_impostor, "pair_genuine": pair_genuine}
    histogram_rows = [{"kind": kind, "lower": float(edges[i]), "upper": float(edges[i + 1]),
                       "count": int(count)} for kind, hist in histograms.items() for i, count in enumerate(hist)]
    write_csv(config["paths"]["h_histogram"], histogram_rows)
    result = {"settings": settings, "operating_points": operating_points,
              "recovery": recovery, "usable_H_identities": len(set(identities)),
              "usable_H_images": len(h), "system_genuine_trials": len(genuine),
              "scope": "H only; all remaining H images are impostor probes; all unique different-identity H pairs",
              "uncertainty": "Correlated repeated galleries and image pairs; rare-rate resolution; no bootstrap or confidence intervals."}
    write_json(config["paths"]["calibration"], result)
    print(result)
    return result


if __name__ == "__main__":
    calibrate(load_config())
