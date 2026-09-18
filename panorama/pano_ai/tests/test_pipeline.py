import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from panorama.pano_ai.pipeline import CorrectionPipeline


def test_pipeline_requires_learned_checkpoints():
    try:
        CorrectionPipeline(toggles={"dots": False, "sharpen": False}, checkpoints={}, device="cpu")
    except FileNotFoundError as exc:
        assert "checkpoint" in str(exc).lower()
    else:
        raise AssertionError("enabled learned stages must not run with random weights")


def test_classical_only_pipeline():
    pipeline = CorrectionPipeline(
        toggles={"glare": False, "nadir_zenith": False, "color": False, "dots": True, "sharpen": True},
        device="cpu",
    )
    image = np.full((64, 128, 3), 128, dtype=np.uint8)
    output = pipeline.run(image)
    assert output.shape == image.shape
    assert output.dtype == np.uint8
