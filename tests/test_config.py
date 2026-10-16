from pathlib import Path

import pytest

from ttd.config import load_config

ROOT = Path(__file__).parents[1]


def test_smoke_config_and_override():
    config = load_config(ROOT / "configs" / "smoke.yaml", ["distill.ipc=3"])
    assert config.distill.ipc == 3
    assert config.trajectory.timesteps == [9, 4, 0]


def test_invalid_timestep_order():
    with pytest.raises(ValueError, match="descending"):
        load_config(ROOT / "configs" / "smoke.yaml", ["trajectory.timesteps=[0, 4, 9]"])
