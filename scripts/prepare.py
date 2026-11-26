"""Prepare the LFW index and embeddings."""
from pathlib import Path

from src.common import load_config
from src.data.prepare import build_index
from src.embedding.extract import extract


def main():
    config = load_config()
    if not Path(config["paths"]["index"]).exists():
        build_index(config)
    extract(config)


if __name__ == "__main__":
    main()
