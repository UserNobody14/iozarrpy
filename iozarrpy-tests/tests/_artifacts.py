from __future__ import annotations

import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output-datasets"


def dataset_path(name: str) -> str:
    """
    Return a stable dataset path under iozarrpy-tests/tests/output-datasets and ensure it is empty.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    if path.exists():
        shutil.rmtree(path)
    return str(path)


