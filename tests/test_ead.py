"""EAD model: ion types and precursor / charge-reduced precursor removal."""

import hashlib

import numpy as np
import pytest

from ms2pip import exceptions
from ms2pip._spectrum_processing import _preprocess_spectrum
from ms2pip._utils.ion_types import parse_ion_type
from ms2pip._utils.xgb_models import validate_model
from ms2pip.constants import MODELS
from ms2pip.spectrum import ObservedSpectrum

PROTON = 1.007276
H = 1.007825
NEUTRON = 1.003355

# [M+2H]2+ of a 1000 Da peptide, and the peaks EAD leaves behind from it.
M = 1000.0
MH2 = M + 2 * PROTON
PRECURSOR_MZ = MH2 / 2
FRAGMENT_MZ = 400.2
PRECURSOR_PEAKS = [
    PRECURSOR_MZ,  # intact [M+2H]2+
    PRECURSOR_MZ + NEUTRON / 2,  # its first isotope
    MH2,  # charge-reduced [M+2H]+.
    MH2 + H,  # charge-reduced with hydrogen gain
    MH2 - 2 * H + NEUTRON,  # charge-reduced, hydrogen loss, isotope
]


def _spectrum(precursor_charge=2):
    mz = np.array(sorted([FRAGMENT_MZ, *PRECURSOR_PEAKS]))
    return ObservedSpectrum(
        mz=mz,
        intensity=np.full(len(mz), 10.0),
        precursor_mz=PRECURSOR_MZ,
        precursor_charge=precursor_charge,
    )


def test_ead_model_ion_types():
    assert MODELS["EAD"]["fragmentation"] == "ead_ext"
    parsed = [parse_ion_type(it)[:3] for it in MODELS["EAD"]["ion_types"]]
    assert parsed == [("b", 0, 1), ("y", 0, 1), ("c", 0, 1), ("z", 1, 1)]


@pytest.mark.parametrize("tolerance, mode", [(0.02, "Da"), (25, "ppm")])
def test_remove_charge_reduced_precursor(tolerance, mode):
    spectrum = _spectrum()
    spectrum.remove_charge_reduced_precursor(tolerance, mode)
    kept = spectrum.mz[spectrum.intensity > 0]
    np.testing.assert_allclose(kept, [FRAGMENT_MZ])


def test_remove_charge_reduced_precursor_without_charge_only_removes_precursor():
    spectrum = _spectrum(precursor_charge=None)
    spectrum.remove_charge_reduced_precursor(0.02, "Da")
    removed = spectrum.mz[spectrum.intensity == 0]
    np.testing.assert_allclose(removed, [PRECURSOR_MZ])


def test_preprocess_ead_normalises_without_precursor():
    spectrum = _spectrum()
    _preprocess_spectrum(spectrum, "EAD", 0.02, "Da")
    fragment = spectrum.intensity[np.isclose(spectrum.mz, FRAGMENT_MZ)]
    # The fragment is the only peak left, so it holds the whole normalised intensity.
    np.testing.assert_allclose(2**fragment - 0.001, [1.0])


def test_preprocess_other_models_keep_precursor():
    spectrum = _spectrum()
    _preprocess_spectrum(spectrum, "HCD", 0.02, "Da")
    assert np.all(2**spectrum.intensity - 0.001 > 0)


def test_validate_model_without_trained_files_raises(monkeypatch):
    monkeypatch.setitem(MODELS, "UNTRAINED", {"ion_types": ["C"], "fragmentation": "ead"})
    with pytest.raises(exceptions.UntrainedModelError, match="UNTRAINED"):
        validate_model("UNTRAINED")


def test_ead_model_has_a_hashed_file_per_ion_type():
    files = MODELS["EAD"]["xgboost_model_files"]
    assert sorted(files) == sorted(it.lower() for it in MODELS["EAD"]["ion_types"])
    assert all(len(MODELS["EAD"]["model_hash"][f]) == 40 for f in files.values())


def _local_only_model(monkeypatch, model_dir, content=b"model bytes"):
    (model_dir / "local_C.xgboost").write_bytes(content)
    monkeypatch.setitem(MODELS, "LOCAL", {
        "ion_types": ["C"], "fragmentation": "ead",
        "xgboost_model_files": {"c": "local_C.xgboost"},
        "model_hash": {"local_C.xgboost": hashlib.sha1(b"model bytes").hexdigest()},
        "download_url": None,
    })


def _no_download(*args, **kwargs):
    raise AssertionError("a local-only model must not be downloaded")


def test_local_only_model_is_used_from_model_dir(monkeypatch, tmp_path):
    _local_only_model(monkeypatch, tmp_path)
    monkeypatch.setattr("urllib.request.urlretrieve", _no_download)
    assert validate_model("LOCAL", tmp_path) == tmp_path


@pytest.mark.parametrize("content", [None, b"other bytes"])
def test_local_only_model_missing_or_changed_raises_without_download(monkeypatch, tmp_path, content):
    _local_only_model(monkeypatch, tmp_path)
    if content is None:
        (tmp_path / "local_C.xgboost").unlink()
    else:
        (tmp_path / "local_C.xgboost").write_bytes(content)
    monkeypatch.setattr("urllib.request.urlretrieve", _no_download)
    with pytest.raises(exceptions.ModelFileNotFoundError, match="local_C.xgboost"):
        validate_model("LOCAL", tmp_path)


def test_model_with_download_url_is_fetched_from_it(monkeypatch, tmp_path):
    hosted = tmp_path / "hosted"
    hosted.mkdir()
    _local_only_model(monkeypatch, hosted)
    MODELS["LOCAL"]["download_url"] = hosted.as_uri()
    model_dir = tmp_path / "models"
    assert validate_model("LOCAL", model_dir) == model_dir
    assert (model_dir / "local_C.xgboost").read_bytes() == b"model bytes"
