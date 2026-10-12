from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from ms2pip._utils.xgb_models import load_internal_xgb_models, load_xgb_models
from ms2pip.constants import MODELS
from ms2pip.core import correlate
from ms2pip.result import mask_low_mz, mask_mz

DATA = Path(__file__).parent / "test_data"


@pytest.fixture(autouse=True)
def _hashless_astral(monkeypatch):
    """The dummy boosters below are not the released files: validate them as hashless."""
    monkeypatch.setitem(MODELS["Astral"], "model_hash", {})


def _dummy_models(tmp_path, value=-5.0):
    m = MODELS["Astral"]
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def _correlate_small(model_dir, **kwargs):
    return correlate(DATA / "astral_small.tsv", DATA / "astral_small.mgf", psm_filetype="tsv", model="Astral",
                     model_dir=model_dir, ms2_tolerance=10.0, ms2_tolerance_mode="ppm", **kwargs)


def _assert_identical(a, b, path="result"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _assert_identical(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype, path
        assert np.array_equal(a, b), path
    else:
        assert a == b, path


def _arrays(r):
    return {"theoretical_mz": r.theoretical_mz, "predicted": r.predicted_intensity,
            "observed": r.observed_intensity, "internal": r.internal}


def _preloaded(model_dir):
    return {"terminal": load_xgb_models(MODELS["Astral"], model_dir),
            "internal": load_internal_xgb_models(MODELS["Astral"], model_dir)}


def test_correlate_preloaded_terminal_internal_identical(tmp_path):
    model_dir = _dummy_models(tmp_path)
    loaded = _correlate_small(model_dir)
    preloaded = _correlate_small(model_dir, xgb_models=_preloaded(model_dir))
    assert len(loaded) == len(preloaded) and any(r.internal for r in loaded)
    for a, b in zip(loaded, preloaded):
        _assert_identical(_arrays(a), _arrays(b))


def test_correlate_uses_preloaded_boosters_not_files(tmp_path):
    """Boosters passed in (-3.0) differ from the files in model_dir (-5.0): a reload would show."""
    model_dir = _dummy_models(tmp_path / "disk")
    other = _dummy_models(tmp_path / "preloaded", value=-3.0)
    results = _correlate_small(model_dir, xgb_models=_preloaded(other))
    internal = [r for r in results if r.internal]
    assert internal
    for r in internal:
        for key in ("yb", "ya"):
            assert np.all(r.internal["predicted"][key] == np.float32(-3.0))
    for r in results:
        if r.predicted_intensity:
            assert all(np.all(v == np.float32(-3.0)) for v in r.predicted_intensity.values())


def test_correlate_plain_dict_is_terminal(tmp_path):
    """A plain dict stays the terminal boosters (as upstream); internal ones come from model_dir."""
    model_dir = _dummy_models(tmp_path / "disk")
    other = _dummy_models(tmp_path / "preloaded", value=-3.0)
    results = _correlate_small(model_dir, xgb_models=load_xgb_models(MODELS["Astral"], other))
    r = next(r for r in results if r.internal)
    assert np.all(r.predicted_intensity["y"] == np.float32(-3.0))
    assert np.all(r.internal["predicted"]["yb"] == np.float32(-5.0))
