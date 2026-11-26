"""Run the fixed-photo verification study using the current config."""
from pathlib import Path

from src.common import load_config
from src.evaluation.gallery import evaluate


def main():
    config = load_config()
    for key in ("index", "embeddings", "calibration"):
        path = Path(config["paths"][key])
        if not path.is_file():
            raise SystemExit(f"Missing {key}: {path}. Complete preparation and H calibration first.")
    print(evaluate(config))


if __name__ == "__main__":
    main()
