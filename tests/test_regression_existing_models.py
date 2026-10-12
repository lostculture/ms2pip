"""Existing models must give byte-identical outputs after the Astral changes."""

from pathlib import Path

import numpy as np
import pytest

from ms2pip.constants import MODELS
from ms2pip.core import correlate, get_training_data

DATA = Path(__file__).parent / "test_data"
MODEL_DIR = Path.home() / ".ms2pip"


def _model_files_present(model):
    files = MODELS[model]["xgboost_model_files"].values()
    return all((MODEL_DIR / f).exists() for f in files)


@pytest.mark.parametrize("model", ["HCD2021", "EAD"])
def test_existing_model_unchanged(model):
    if not _model_files_present(model):
        pytest.skip(f"{model} model files not available")
    ref = np.load(DATA / f"regression_{model.lower()}.npz")
    assert not np.isnan(ref["corr"]).any()
    kw = dict(psm_filetype="tsv", model=model, ms2_tolerance=10.0, ms2_tolerance_mode="ppm")
    res = correlate(DATA / "astral_small.tsv", DATA / "astral_small.mgf", compute_correlations=True, **kw)
    keys = set()
    for r in res:
        assert r.internal is None
        for k, v in (r.predicted_intensity or {}).items():
            keys.add(f"pred_{r.psm_index}_{k}")
            np.testing.assert_array_equal(v, ref[f"pred_{r.psm_index}_{k}"])
        for k, v in (r.observed_intensity or {}).items():
            keys.add(f"obs_{r.psm_index}_{k}")
            np.testing.assert_array_equal(v, ref[f"obs_{r.psm_index}_{k}"])
    assert keys == {k for k in ref.files if k.startswith(("pred_", "obs_"))}
    np.testing.assert_array_equal([r.correlation for r in sorted(res, key=lambda r: r.psm_index)], ref["corr"])
    td = get_training_data(DATA / "astral_small.tsv", DATA / "astral_small.mgf", **kw)
    np.testing.assert_array_equal(td.to_numpy(dtype=float), ref["train"])
