"""Read images, extract embeddings, and assign G/P/H roles."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.common import validate_roles, write_csv, write_json


def assign_initial_groups(rows, split):
    by_identity = defaultdict(list)
    for row in rows:
        by_identity[row["identity_id"]].append(row)
    rng = np.random.default_rng(split["seed"])
    identities = rng.permutation(sorted(by_identity)).tolist()
    eligible = [name for name in identities if
                sum(r["valid"] == "true" for r in by_identity[name]) >= 2]
    if len(eligible) < split["G"] or len(identities) < sum(split[g] for g in ("G", "P", "H")):
        raise ValueError("Not enough readable identities for the configured split.")
    gallery = set(eligible[:split["G"]])
    remaining = [name for name in identities if name not in gallery]
    pool = set(remaining[:split["P"]])
    calibration = set(remaining[split["P"]:split["P"] + split["H"]])
    for name, images in by_identity.items():
        group = "G" if name in gallery else "P" if name in pool else "H" if name in calibration else "D"
        for row in images:
            row["partition"] = group
    assign_roles(rows, final=False)
    validate_roles(rows, split)


def assign_roles(rows, final):
    usable = defaultdict(list)
    for row in rows:
        if row["partition"] == "unused":
            row["partition"] = "D"
        ok = row["embedding_status"] == "ok" if final else row["valid"] == "true"
        row["role"] = "unavailable" if not ok else {
            "P": "pool", "H": "calibration", "D": "development", "G": "held_out"
        }[row["partition"]]
        if ok and row["partition"] == "G":
            usable[row["identity_id"]].append(row)
    for images in usable.values():
        min(images, key=lambda r: r["image_id"])["role"] = "enrollment"


def finalize_roles(rows, split):
    by_identity = defaultdict(list)
    for row in rows:
        by_identity[row["identity_id"]].append(row)
    usable_counts = {name: sum(r["embedding_status"] == "ok" for r in images)
                     for name, images in by_identity.items()}
    broken = sorted(name for name, images in by_identity.items()
                    if images[0]["partition"] == "G" and usable_counts[name] < 2)
    replacements = sorted(name for name, images in by_identity.items()
                          if images[0]["partition"] == "P" and usable_counts[name] >= 2)
    if len(broken) > len(replacements):
        raise ValueError("Not enough P identities to repair G without changing H.")
    swaps = []
    for removed, added in zip(broken, replacements):
        for row in by_identity[removed]:
            row["partition"] = "P"
        for row in by_identity[added]:
            row["partition"] = "G"
        swaps.append({"removed_G": removed, "added_G": added})
    assign_roles(rows, final=True)
    validate_roles(rows, split, final=True)
    return swaps


def build_index(config):
    root = Path(config["paths"]["image_root"])
    destination = Path(config["paths"]["index"])
    if destination.exists():
        raise FileExistsError("Index exists; choose a fresh work directory/config for a new preparation.")
    rows = []
    for path in sorted(root.glob("*/*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        error = ""
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()
        except (OSError, ValueError) as exc:
            error = type(exc).__name__
        if path.stem.rsplit("_", 1)[0] != path.parent.name:
            raise ValueError(f"LFW filename/identity mismatch: {path}")
        rows.append({"image_id": path.stem, "identity_id": path.parent.name,
                     "relative_path": path.relative_to(root).as_posix(),
                     "valid": "false" if error else "true", "partition": "", "role": "",
                     "embedding_status": "unreadable" if error else "pending",
                     "embedding_row": -1, "error": error})
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("No LFW images found, or duplicate image IDs; check image_root.")
    assign_initial_groups(rows, config["split"])
    write_csv(destination, rows)
    metadata = {"dataset": config["dataset"], "split_seed": config["split"]["seed"],
                "counts": {"images": len(rows), "identities": len({r["identity_id"] for r in rows}),
                           "readable_images": sum(r["valid"] == "true" for r in rows)},
                "partition_counts": validate_roles(rows, config["split"]),
                "embedding_state": "pending"}
    write_json(config["paths"]["metadata"], metadata)
    print(metadata)
