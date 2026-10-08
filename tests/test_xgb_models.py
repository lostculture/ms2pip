"""Model loading: verified-file cache, CUDA_VISIBLE_DEVICES workaround, preloaded models in correlate."""

import os

import numpy as np
import pytest
from ms2rescore_rs import MS2Spectrum, Precursor
from psm_utils import PSM, Peptidoform

from ms2pip import core
from ms2pip._utils import xgb_models


@pytest.fixture
def model_file(tmp_path, monkeypatch):
    monkeypatch.setattr(xgb_models, "_verified_files", set())
    path = tmp_path / "model_B.xgboost"
    path.write_bytes(b"model")
    return path


def test_model_hash_checked_once_per_file(model_file, monkeypatch):
    calls = []

    def check(filename, model_hash):
        calls.append(filename)
        return True

    monkeypatch.setattr(xgb_models, "_check_model_integrity", check)
    for _ in range(3):
        assert xgb_models._check_model_presence(model_file.name, "hash", model_file.parent)
    assert len(calls) == 1


def test_changed_model_file_is_checked_again(model_file, monkeypatch):
    calls = []
    monkeypatch.setattr(xgb_models, "_check_model_integrity", lambda f, h: calls.append(f) or True)
    xgb_models._check_model_presence(model_file.name, "hash", model_file.parent)
    model_file.write_bytes(b"another model")
    xgb_models._check_model_presence(model_file.name, "hash", model_file.parent)
    assert len(calls) == 2


def test_failed_hash_is_not_cached(model_file, monkeypatch):
    calls = []
    monkeypatch.setattr(xgb_models, "_check_model_integrity", lambda f, h: calls.append(f) and False)
    assert not xgb_models._check_model_presence(model_file.name, "hash", model_file.parent)
    assert not xgb_models._check_model_presence(model_file.name, "hash", model_file.parent)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "value, kept",
    [("-1", False), (" -1", False), ("0", True), ("1", True), ("GPU-1234", True), (None, None)],
)
def test_cuda_visible_devices_only_dropped_when_hiding_all(monkeypatch, value, kept):
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    xgb_models._drop_cuda_visible_devices_if_hiding_all()
    if kept is None:
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    else:
        assert ("CUDA_VISIBLE_DEVICES" in os.environ) is kept


@pytest.mark.parametrize("preloaded", [None, {"b": "booster B", "y": "booster Y"}])
def test_correlate_passes_preloaded_models(monkeypatch, tmp_path, preloaded):
    """correlate(..., xgb_models=m) must hand m to predict_intensities instead of reloading."""
    seen = []

    def fake_predict(features, num_ions, model_params, model_dir, processes=None, xgb_models=None):
        seen.append(xgb_models)
        return [{"b": np.zeros(n, np.float32), "y": np.zeros(n, np.float32)} for n in num_ions]

    monkeypatch.setattr(core, "validate_model", lambda model, model_dir=None: tmp_path)
    monkeypatch.setattr(core, "predict_intensities", fake_predict)
    psm = PSM(
        peptidoform=Peptidoform("ACDEFK/2"),
        spectrum_id="test_spectrum",
        spectrum=MS2Spectrum(
            identifier="test_spectrum",
            mz=[175.054, 276.167, 488.319],
            intensity=[100.0, 200.0, 300.0],
            precursor=Precursor(mz=341.65, charge=2, rt=100.0),
        ),
    )
    results = core.correlate([psm], model="HCD", xgb_models=preloaded)
    assert len(results) == 1 and results[0].predicted_intensity is not None
    assert seen == [preloaded]
