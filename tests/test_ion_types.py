"""Ion-type keys with hydrogen-shift variants (z+1, z+1_2, c-1, ...)."""

import numpy as np
import pytest
from psm_utils import PSM, Peptidoform

from ms2pip._utils.ion_types import (
    annotation_label,
    internal_label,
    is_c_terminal,
    is_internal,
    parse_ion_type,
)
from ms2pip._utils.xgb_models import predict_intensities
from ms2pip.result import ProcessingResult


@pytest.mark.parametrize(
    "key, expected",
    [
        ("b", ("b", 0, 1, "")),
        ("y2", ("y", 0, 2, "")),
        ("z+1", ("z", 1, 1, "")),
        ("z+2", ("z", 2, 1, "")),
        ("z+1_2", ("z", 1, 2, "")),
        ("c-1", ("c", -1, 1, "")),
        ("C-1", ("c", -1, 1, "")),
    ],
)
def test_parse_ion_type(key, expected):
    assert parse_ion_type(key) == expected


@pytest.mark.parametrize("bad", ["", "q", "z+", "z+_2", "z+1x", "z1_2", "z+0", "z0"])
def test_parse_ion_type_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_ion_type(bad)


@pytest.mark.parametrize(
    "key, c_term",
    [("b", False), ("b2", False), ("c-1", False), ("y", True), ("y2", True), ("z+1_2", True)],
)
def test_is_c_terminal(key, c_term):
    assert is_c_terminal(key) is c_term


class _CountingBooster:
    """Stand-in for xgb.Booster: predicts 0..n-1 so ordering is visible."""

    def predict(self, dmatrix):
        return np.arange(dmatrix.num_row(), dtype=np.float32)


def test_predict_intensities_orders_variant_series():
    features = np.zeros((3, 4), dtype=np.float32)
    preds = predict_intensities(
        features,
        num_ions=[3],
        model_params={},
        model_dir=None,
        xgb_models={"c-1": _CountingBooster(), "z+1": _CountingBooster()},
    )[0]
    np.testing.assert_array_equal(preds["c-1"], [0, 1, 2])
    np.testing.assert_array_equal(preds["z+1"], [2, 1, 0])


def test_training_data_reverses_c_terminal_variants(monkeypatch):
    from ms2pip import core
    from ms2pip._utils.feature_names import get_feature_names

    monkeypatch.setitem(core.MODELS, "TEST_EAD", {"ion_types": ["c-1", "z+1"]})
    n_features = len(get_feature_names())
    result = ProcessingResult(
        psm_index=7,
        psm=PSM(peptidoform=Peptidoform("PEPK/2"), spectrum_id="0"),
        feature_vectors=np.zeros((3, n_features), dtype=np.float32),
        observed_intensity={"c-1": np.array([1.0, 2.0, 3.0]), "z+1": np.array([1.0, 2.0, 3.0])},
    )
    data = core._assemble_training_data([result], "TEST_EAD")
    assert list(data["target_c-1"]) == [1.0, 2.0, 3.0]
    assert list(data["target_z+1"]) == [3.0, 2.0, 1.0]


def test_spectronaut_fragment_charge_and_variants():
    from ms2pip.spectrum_output import Spectronaut

    result = ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform=Peptidoform("PEPK/2"), spectrum_id="0"),
        theoretical_mz={
            "y2": np.array([74.1, 122.6, 171.1]),
            "z+1": np.array([131.1, 228.1, 357.2]),
        },
        predicted_intensity={"y2": np.zeros(3), "z+1": np.zeros(3)},
    )
    rows = list(Spectronaut._yield_fragment_info(result))
    assert {r["FragmentCharge"] for r in rows if r["FragmentType"] == "y"} == {"2"}
    # Spectronaut has no notation for hydrogen-shift variants: z+1 must not be
    # written as a plain z with charge "+1".
    assert not [r for r in rows if r["FragmentType"] == "z"]


def test_as_spectra_annotates_variant_ions():
    result = ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform=Peptidoform("PEPK/2"), spectrum_id="0"),
        theoretical_mz={
            "z": np.array([130.1, 227.1, 356.2]),
            "z+1_2": np.array([66.1, 114.6, 179.1]),
        },
        predicted_intensity={"z": np.zeros(3), "z+1_2": np.zeros(3)},
    )
    predicted, _ = result.as_spectra()
    assert sorted(predicted.annotations) == sorted(
        ["z1", "z2", "z3", "z1+1^2", "z2+1^2", "z3+1^2"]
    )


@pytest.mark.parametrize("key, parsed", [
    ("b-h2o", ("b", 0, 1, "h2o")), ("y-nh3", ("y", 0, 1, "nh3")), ("y2-h2o", ("y", 0, 2, "h2o")),
    ("a", ("a", 0, 1, "")), ("z+1", ("z", 1, 1, "")), ("y2", ("y", 0, 2, "")),
])
def test_parse_loss_keys(key, parsed):
    assert parse_ion_type(key) == parsed


@pytest.mark.parametrize("bad", ["b-co", "z+1-h2o", "yb2", "b-"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_ion_type(bad)


def test_internal_keys():
    assert is_internal("yb") and is_internal("YA")
    assert not is_internal("y") and not is_internal("b-h2o")
    assert not is_c_terminal("yb")
    assert is_c_terminal("y-nh3") and not is_c_terminal("a-h2o")
    assert annotation_label("b-h2o", 3) == "b3-H2O"
    assert internal_label("yb", 3, 5) == "yb[3-5]"
