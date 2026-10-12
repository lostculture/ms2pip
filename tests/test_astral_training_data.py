from pathlib import Path

import numpy as np
import pandas as pd

from ms2pip.core import get_training_data
from ms2pip._utils.internal import internal_feature_names
from ms2pip.constants import MODELS

DATA = Path(__file__).parent / "test_data"


def _tables():
    return get_training_data(DATA / "astral_small.tsv", DATA / "astral_small.mgf", psm_filetype="tsv",
                             model="Astral", ms2_tolerance=10.0, ms2_tolerance_mode="ppm")


def test_two_tables_with_mz_and_mask():
    terminal, internal = _tables()
    keys = [k.lower() for k in MODELS["Astral"]["ion_types"]]
    for k in keys:
        assert f"target_{k}" in terminal and f"mz_{k}" in terminal
        low = (terminal[f"mz_{k}"] < 150.0) | (terminal[f"mz_{k}"] == 0) | (terminal[f"mz_{k}"] > 2000.0)
        assert terminal.loc[low, f"target_{k}"].isna().all()
        assert terminal.loc[~low, f"target_{k}"].notna().all()
    assert list(internal.columns[:4]) == ["psm_index", "s", "e", "n_isobaric"]
    assert set(internal_feature_names()) <= set(internal.columns)
    assert internal.loc[internal.mz_yb < 150, "target_yb"].isna().all()
    for key in ("yb", "ya"):
        assert internal.loc[internal[f"mz_{key}"] > 2000.0, f"target_{key}"].isna().all()
    assert internal.psm_index.nunique() == terminal.psm_index.nunique() == 3


def test_observed_internal_peak_is_a_target():
    _, internal = _tables()
    assert (internal.target_yb > -9.9).sum() > 0  # some yb observed in real Astral spectra


def test_hcd2021_still_one_table():
    out = get_training_data(DATA / "astral_small.tsv", DATA / "astral_small.mgf", psm_filetype="tsv",
                            model="HCD2021", ms2_tolerance=10.0, ms2_tolerance_mode="ppm")
    assert isinstance(out, pd.DataFrame)
    assert not any(c.startswith("mz_") for c in out.columns)


def test_outside_mz_range_synthetic():
    # The 3-PSM fixture has no fragment above 2000, so exercise the masking helper directly.
    from ms2pip.core import _outside_mz_range

    mz = np.array([0.0, 140.0, 400.0, 2100.0])
    assert _outside_mz_range(mz, 150.0, 2000.0).tolist() == [True, True, False, True]
    assert _outside_mz_range(mz, 150.0, None).tolist() == [True, True, False, False]
