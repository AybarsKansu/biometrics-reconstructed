"""Offline evaluation of fixed photo embeddings against nested enrollment galleries."""
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageStat

from src.common import load_data, read_json, write_csv, write_json


def clustered_rate(events, identities, repetitions, seed):
    """Image-weighted rate; resample whole probe identities for a fixed gallery."""
    groups, inverse = np.unique(identities, return_inverse=True)
    totals = np.bincount(inverse)
    errors = np.bincount(inverse, weights=events)
    count = int(errors.sum())
    result = {"events": count, "images": len(events), "identities": len(groups),
              "rate": count / len(events), "ci95_low": None, "ci95_high": None}
    if len(groups) < 2 or np.allclose(errors / totals, count / len(events)):
        # A degenerate empirical bootstrap gives no information on unobserved errors.
        return result
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repetitions, 128):
        draw = rng.integers(len(groups), size=(min(128, repetitions - start), len(groups)))
        samples.extend(errors[draw].sum(axis=1) / totals[draw].sum(axis=1))
    result["ci95_low"], result["ci95_high"] = map(float, np.quantile(samples, [0.025, 0.975]))
    return result


def score_probes(vectors, probe_indices, gallery_indices, block_size):
    gallery = vectors[gallery_indices].T.copy(order="F")
    probes = vectors[probe_indices]
    # Warm up the matrix operation. Report batched scoring time per probe.
    np.matmul(probes[:min(len(probes), block_size)], gallery)
    scores, nearest = [], []
    elapsed = 0.0
    for start in range(0, len(probes), block_size):
        block = probes[start:start + block_size]
        tick = perf_counter()
        similarities = block @ gallery
        best = similarities.argmax(axis=1)
        maxima = similarities[np.arange(len(block)), best]
        elapsed += perf_counter() - tick
        scores.extend(np.clip(maxima, -1, 1))
        nearest.extend(best)
    return np.array(scores), np.array(nearest), 1000 * elapsed / len(probes)


def photo_profile(root, row):
    try:
        with Image.open(root / row["relative_path"]) as image:
            width, height = image.size
            brightness = ImageStat.Stat(image.convert("L")).mean[0]
    except (OSError, ValueError):
        return "unavailable", "", "", ""
    band = "dark" if brightness < 85 else "mid" if brightness < 170 else "bright"
    return band, width, height, round(brightness, 2)


def evaluate(config):
    _, vectors, rows = load_data(config)
    paths, settings = config["paths"], config["evaluation"]
    calibrated = read_json(paths["calibration"])
    sizes = sorted(set(settings["gallery_sizes"]))
    reps, block, seed = (settings[k] for k in ("bootstrap_repetitions", "block_size", "seed"))
    enrollment = {r["identity_id"]: i for i, r in enumerate(rows)
                  if r["partition"] == "G" and r["role"] == "enrollment"}
    if not sizes or sizes[0] < 1 or sizes[-1] > len(enrollment) or block < 1 or reps < 2:
        raise ValueError("Check gallery sizes, available enrollments, block size and bootstrap repetitions.")
    names = np.random.default_rng(seed).permutation(sorted(enrollment))
    common_names = set(names[:sizes[0]])
    known = [i for i, r in enumerate(rows) if r["role"] == "held_out" and
             r["partition"] == "G" and r["identity_id"] in common_names]
    unknown = [i for i, r in enumerate(rows) if r["partition"] in {"D", "unused"}]
    if not known or not unknown:
        raise ValueError("Evaluation needs usable held-out gallery images and separate development images.")
    probes = known + unknown
    ids = np.array([rows[i]["identity_id"] for i in probes])
    points = calibrated["operating_points"]
    if not points:
        raise ValueError("No calibrated operating points.")
    threshold_rows, benchmark, error_rows, strata = [], [], [], []
    error_fields = ["target_fmr", "error", "image_id", "relative_path", "score",
                    "brightness_band", "width", "height", "mean_luminance"]
    profiles = {}
    for size in sizes:
        indices = [enrollment[name] for name in names[:size]]
        scores, nearest, runtime = score_probes(vectors, probes, indices, block)
        correct = names[nearest[:len(known)]] == ids[:len(known)]
        benchmark.append({"gallery_size": size, "known_probes": len(known), "unknown_probes": len(unknown),
                          "rank1_accuracy": float(correct.mean()), "scoring_ms_per_probe": runtime})
        for target, point in points.items():
            threshold = point["threshold"]
            accepted = scores >= threshold
            known_accept = accepted[:len(known)]
            masks = [("D", "false_accept", accepted[len(known):], unknown, ids[len(known):]),
                     ("G", "false_reject", ~known_accept, known, ids[:len(known)]),
                     ("G", "wrong_identity_accepted", known_accept & ~correct, known, ids[:len(known)])]
            for group, event, mask, positions, people in masks:
                threshold_rows.append({"gallery_size": size, "calibration_target_fmr": target,
                                       "threshold": threshold, "group": group, "event": event,
                                       **clustered_rate(mask, people, reps, seed)})
                if size != sizes[-1]:
                    continue
                counts = defaultdict(Counter)
                for j, (pos, is_error) in enumerate(zip(positions, mask)):
                    row = rows[pos]
                    if pos not in profiles:
                        profiles[pos] = photo_profile(Path(paths["image_root"]), row)
                    band, w, h, light = profiles[pos]
                    counts[band]["images"] += 1
                    counts[band]["events"] += int(is_error)
                    if is_error:
                        score = scores[j + (len(known) if group == "D" else 0)]
                        error_rows.append({"target_fmr": target, "error": event, "image_id": row["image_id"],
                                           "relative_path": row["relative_path"], "score": float(score),
                                           "brightness_band": band, "width": w, "height": h,
                                           "mean_luminance": light})
                for band, count in sorted(counts.items()):
                    strata.append({"gallery_size": size, "target_fmr": target, "event": event,
                                   "brightness_band": band, **count, "rate": count["events"] / count["images"]})
    out = Path(paths["results"]) / "verification"
    write_csv(out / "thresholds.csv", threshold_rows)
    write_csv(out / "gallery_sizes.csv", benchmark)
    write_csv(out / "errors.csv", error_rows, error_fields)
    write_csv(out / "brightness.csv", strata)
    write_json(out / "protocol.json", {"seed": seed, "gallery_sizes": sizes, "bootstrap_repetitions": reps,
               "calibration_gallery_size": calibrated["settings"]["gallery_size"],
               "known_probe_identities": len(common_names), "known_probes": len(known),
               "development_identities_with_vectors": len(set(ids[len(known):])),
               "development_probes": len(unknown)})
    return out
