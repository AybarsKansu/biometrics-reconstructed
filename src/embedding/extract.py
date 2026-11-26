"""Extract normalized face embeddings and finalize enrollment roles."""
from collections import Counter
from pathlib import Path

import numpy as np

from src.common import check_embeddings, normalize, read_index, read_json, validate_roles, write_csv, write_json
from src.data.prepare import finalize_roles


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
