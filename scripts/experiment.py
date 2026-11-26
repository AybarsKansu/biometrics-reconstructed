"""Run an explicitly selected pilot, or opt into the 18-setting initial batch."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.attacks.attacks import candidates
from src.common import (read_json, load_config,
                    load_data, select, write_json)
from src.verifier.comparator import run_loop


def experiment_data(vectors, rows):
    gallery, gallery_rows = select(vectors, rows, role="enrollment")
    pool, _ = select(vectors, rows, partition="P")
    held_out_lists = defaultdict(list)
    for vector, row in zip(vectors, rows):
        if row["role"] == "held_out":
            held_out_lists[row["identity_id"]].append(vector)
    held_out = {name: np.asarray(items, dtype=np.float32) for name, items in held_out_lists.items()}
    identities = [row["identity_id"] for row in gallery_rows]
    if not all(name in held_out for name in identities):
        raise ValueError("Enrollment identity lacks private held-out embeddings.")
    return gallery, identities, held_out, pool


def run(config, methods, granularities, targets, include_zero=True, label="pilot"):
    calibration = read_json(config["paths"]["calibration"])
    _, vectors, rows = load_data(config)
    gallery, identities, held_out, pool = experiment_data(vectors, rows)
    if len(gallery) != config["calibration"]["gallery_size"]:
        raise ValueError("Surrogate gallery size must equal the G gallery size.")
    if any(granularity not in ("float", "8", "4") for granularity in granularities):
        raise ValueError("1-bit comparator exists; 1-bit attack experiments are deferred.")
    if not methods or any(method not in ("hill", "simplex", "genetic") for method in methods):
        raise ValueError("Choose hill, simplex, or genetic.")
    settings = config["experiment"]
    if settings["query_budget"] < 1 or settings["repetitions"] < 1:
        raise ValueError("Budget and repetition count must be positive.")
    directory = Path(config["paths"]["results"]) / (label + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = {"configuration": config, "label": label,
                "methods": methods, "granularities": granularities, "targets": targets,
                "baseline_reused_across_feedback": include_zero,
                "calibration": calibration}
    write_json(directory / "resolved.json", snapshot)
    all_records = []
    with (directory / "private_records.jsonl").open("w", encoding="utf-8") as stream:
        for target in targets:
            point = calibration["operating_points"][str(target)]
            combinations = [(method, granularity) for method in methods for granularity in granularities]
            if include_zero:
                combinations.insert(0, ("zero", "independent"))
            for method, granularity in combinations:
                for repetition in range(settings["repetitions"]):
                    seed = settings["seed"] + repetition
                    generator = candidates(method, pool, seed, config)
                    record = run_loop(generator, gallery, identities, held_out,
                                      point["threshold"], point["lo"], calibration["recovery"]["threshold"],
                                      "float" if method == "zero" else granularity, settings["query_budget"])
                    record.update(method=method, granularity=granularity, target_fmr=target,
                                  repetition=repetition, seed=seed, budget=settings["query_budget"])
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    all_records.append(record)
                    print(f"{method}/{granularity}/FMR={target} repeat={repetition}: {record['status']}, queries={record['queries']}", flush=True)
    print(f"Private evaluator records: {directory}")
    return directory, all_records


def main():
    config = load_config()
    settings = config["experiment"]
    run(config, settings["methods"], settings["granularities"], settings["targets"])


if __name__ == "__main__":
    main()
