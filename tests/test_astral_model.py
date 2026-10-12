import pytest

from ms2pip import exceptions
from ms2pip._utils.ion_types import is_internal, parse_ion_type
from ms2pip._utils.xgb_models import load_internal_xgb_models, validate_model
from ms2pip.constants import MODELS


def test_astral_entry():
    m = MODELS["Astral"]
    assert m["fragmentation"] == "astral_hcd"
    assert [k.lower() for k in m["ion_types"]] == [
        "y", "b", "a", "y2", "b2", "y-h2o", "b-h2o", "y-nh3", "b-nh3", "a-nh3", "a-h2o"]
    assert [k.lower() for k in m["internal_ion_types"]] == ["yb", "ya"]
    assert m["internal_max_mass"] == 700.0 and m["min_fragment_mz"] == 150.0
    for k in m["ion_types"]:
        parse_ion_type(k)
        assert k.lower() in m["xgboost_model_files"]
    assert set(m["internal_xgboost_model_files"]) == {"yb", "ya"}
    assert all(is_internal(k) for k in m["internal_xgboost_model_files"])


def test_astral_untrained_until_files_exist(tmp_path):
    with pytest.raises(exceptions.UntrainedModelError):
        validate_model("Astral", tmp_path)


def test_astral_files_present_without_hash_accepted(tmp_path):
    m = MODELS["Astral"]
    for f in [*m["xgboost_model_files"].values(), *m["internal_xgboost_model_files"].values()]:
        (tmp_path / f).write_bytes(b"x")
    assert validate_model("Astral", tmp_path) == tmp_path


def test_astral_one_internal_file_missing_is_untrained(tmp_path):
    m = MODELS["Astral"]
    for f in [*m["xgboost_model_files"].values(), m["internal_xgboost_model_files"]["yb"]]:
        (tmp_path / f).write_bytes(b"x")
    with pytest.raises(exceptions.UntrainedModelError):
        validate_model("Astral", tmp_path)


def test_load_internal_empty_for_other_models(tmp_path):
    assert load_internal_xgb_models(MODELS["HCD2021"], tmp_path) == {}
