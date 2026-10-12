from pathlib import Path

import numpy as np
import xgboost as xgb

from ms2pip.constants import MODELS
from ms2pip.core import correlate
from ms2pip.result import mask_low_mz, mask_mz

DATA = Path(__file__).parent / "test_data"


def _dummy_models(tmp_path, value=-5.0):
    m = MODELS["Astral"]
    for files, width in ((m["xgboost_model_files"], 139), (m["internal_xgboost_model_files"], 294)):
        x = np.zeros((4, width), np.float32)
        booster = xgb.train({"base_score": value, "max_depth": 1}, xgb.DMatrix(x, label=np.full(4, value)), 1)
        for f in files.values():
            booster.save_model(tmp_path / f)
    return tmp_path


def test_mask_excludes_ungenerated():
    v = {"b2": np.array([1.0, 2.0, 3.0], np.float32)}
    mz = {"b2": np.array([0.0, 140.0, 400.0], np.float32)}
    assert mask_low_mz(v, mz, 150.0)["b2"].tolist() == [3.0]


def test_correlate_astral_returns_internal(tmp_path):
    results = correlate(DATA / "astral_small.tsv", DATA / "astral_small.mgf", psm_filetype="tsv", model="Astral",
                        model_dir=_dummy_models(tmp_path), ms2_tolerance=10.0, ms2_tolerance_mode="ppm")
    r = next(r for r in results if r.internal)
    k = len(r.internal["s"])
    assert r.internal["predicted"]["yb"].shape == (k,) == r.internal["observed"]["yb"].shape
    assert np.all(r.internal["predicted"]["yb"] == np.float32(-5.0))
    assert set(r.predicted_intensity) == {k.lower() for k in MODELS["Astral"]["ion_types"]}
    pred, obs = r.as_spectra()
    assert any(a.startswith("yb[") for a in pred.annotations)


def test_correlations_and_batch_predict(tmp_path):
    from ms2pip.core import predict_batch
    from psm_utils.io import read_file

    model_dir = _dummy_models(tmp_path)
    results = correlate(DATA / "astral_small.tsv", DATA / "astral_small.mgf", psm_filetype="tsv", model="Astral",
                        model_dir=model_dir, ms2_tolerance=10.0, ms2_tolerance_mode="ppm",
                        compute_correlations=True)
    assert any(r.correlation_internal is not None for r in results if r.internal)
    psms = read_file(str(DATA / "astral_small.tsv"), filetype="tsv")
    out = predict_batch(psms, model="Astral", model_dir=model_dir)
    r = next(r for r in out if r.internal)
    assert "observed" not in r.internal
    assert np.all(r.internal["predicted"]["ya"] == np.float32(-5.0))


def test_mask_mz_upper_bound():
    mz = {"b2": np.array([0.0, 140.0, 400.0, 2100.0])}
    v = {"b2": np.array([1.0, 2.0, 3.0, 4.0])}
    assert mask_mz(v, mz, 150.0, 2000.0)["b2"].tolist() == [3.0]
    assert mask_mz(v, mz, 150.0)["b2"].tolist() == [3.0, 4.0]
    assert mask_low_mz(v, mz, 150.0)["b2"].tolist() == [3.0, 4.0]
