"""Read images, extract embeddings, and assign G/P/H roles."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from common import (check_embeddings, load_config, normalize, read_index, read_json,
                    validate_roles, write_csv, write_json)


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
        group = "G" if name in gallery else "P" if name in pool else "H" if name in calibration else "unused"
        for row in images:
            row["partition"] = group
    assign_roles(rows, final=False)
    validate_roles(rows, split)


def assign_roles(rows, final):
    usable = defaultdict(list)
    for row in rows:
        ok = row["embedding_status"] == "ok" if final else row["valid"] == "true"
        row["role"] = "unavailable" if not ok else {
            "P": "pool", "H": "calibration", "unused": "unused", "G": "held_out"
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


def extract(config):
    from importlib.metadata import version
    import cv2
    from insightface.app import FaceAnalysis

    paths = config["paths"]
    if Path(paths["embeddings"]).exists():
        raise FileExistsError("Embedding store already exists; use it in the next stages.")
    directory = Path(paths["model_root"]) / "models" / "buffalo_l"
    for name in ("w600k_r50.onnx", "det_10g.onnx"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing local model: {directory / name}")
    rows = read_index(paths["index"])
    model = {key: config["model"][key] for key in
             ("name", "det_size", "det_threshold", "face_selection", "preprocessing")}
    model["insightface_version"] = version("insightface")
    app = FaceAnalysis(name="buffalo_l", root=paths["model_root"],
                       allowed_modules=["detection", "recognition"],
                       providers=config["model"]["providers"])
    app.prepare(ctx_id=-1 if config["model"]["providers"] == ["CPUExecutionProvider"] else 0,
                det_size=tuple(model["det_size"]), det_thresh=model["det_threshold"])
    vectors, ids = [], []
    for number, row in enumerate(rows):
        if row["valid"] != "true":
            continue
        try:
            image = cv2.imread(str(Path(paths["image_root"]) / row["relative_path"]))
            if image is None:
                raise ValueError("image_read_failed")
            faces = app.get(image)
            if not faces:
                raise ValueError("no_face_detected")
            face = min(faces, key=lambda f: (-float(f.det_score), *np.asarray(f.bbox).tolist()))
            vectors.append(normalize(face.embedding))
            ids.append(row["image_id"])
        except (ValueError, RuntimeError, cv2.error) as exc:
            row["embedding_status"] = "failed"
            row["error"] = str(exc)[:200]
        if number % 500 == 0:
            print(f"Processed {number}/{len(rows)} images", flush=True)
    vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, 512)
    check_embeddings(vectors, ids)
    positions = {image_id: i for i, image_id in enumerate(ids)}
    for row in rows:
        position = positions.get(row["image_id"], -1)
        row["embedding_row"] = position
        if position >= 0:
            row["embedding_status"] = "ok"
            row["error"] = ""
    swaps = finalize_roles(rows, config["split"])
    destination = Path(paths["embeddings"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, embeddings=vectors, image_ids=np.asarray(ids, dtype=str))
    write_csv(paths["index"], rows)
    metadata = read_json(paths["metadata"])
    metadata.update({"model": model, "embedding_state": "complete",
                     "failed_embeddings": len(rows) - len(ids), "swaps": swaps,
                     "partition_counts": validate_roles(rows, config["split"], final=True),
                     "usable_image_counts": dict(Counter(r["partition"] for r in rows if r["embedding_status"] == "ok"))})
    write_json(paths["metadata"], metadata)
    print(f"Stored {len(ids)} aligned embeddings; {len(rows) - len(ids)} unavailable; {len(swaps)} G/P swaps.")


def main():
    config = load_config()
    if not Path(config["paths"]["index"]).exists():
        build_index(config)
    extract(config)


if __name__ == "__main__":
    main()
