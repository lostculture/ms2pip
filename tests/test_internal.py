"""Internal fragment features for the Astral model."""

import numpy as np
import pytest
from ms2rescore_rs import ms2pip_compute_features, ms2pip_internal_fragments
from psm_utils import Peptidoform

from ms2pip._utils.internal import (
    DESCRIPTOR_NAMES,
    build_internal_features,
    internal_feature_names,
    oracle_segments,
)

PROTON = 1.007276466812
PEPTIDES = ["PEPTIDEK/2", "PAGAGK/2", "PM[+15.9949]AGSAGK/2", "[+42.0106]-AC[+57.0215]DEFGHIKLMNPQR/3",
            "GGGGGGGGGGK/2", "PEK/2"]


@pytest.mark.parametrize("proforma", PEPTIDES)
def test_rust_segments_equal_python_oracle(proforma):
    residues = Peptidoform(proforma).sequential_theoretical_mass[1:-1]
    s, e, n_iso, mz_yb, _ = ms2pip_internal_fragments([proforma], 700.0)[0]
    assert list(zip(s.tolist(), e.tolist(), n_iso.tolist())) == oracle_segments(residues, 700.0)


def test_feature_shape_names_and_site_rows():
    proforma, sequence = "PEPTIDEK/2", "PEPTIDEK"
    site = ms2pip_compute_features([proforma])[0].reshape(-1, 139)
    s, e, n_iso, mz_yb, _ = ms2pip_internal_fragments([proforma], 700.0)[0]
    x = build_internal_features(site, sequence, s, e, n_iso, mz_yb, precursor_mass=927.45, charge=2)
    names = internal_feature_names()
    assert x.shape == (len(s), len(names)) == (len(s), 2 * 139 + len(DESCRIPTOR_NAMES))
    assert x.dtype == np.float32
    # segment (1,2) = "EP": N-side row s-1 = 0 (cleavage P|E), C-side row e = 2 (cleavage P|T)
    np.testing.assert_array_equal(x[0, :139], site[0])
    np.testing.assert_array_equal(x[0, 139:278], site[2])
    d = dict(zip(DESCRIPTOR_NAMES, x[0, 278:]))
    assert d["seg_len"] == 2 and d["seg_n_p"] == 1 and d["seg_n_e"] == 1
    assert d["seg_start_p"] == 0 and d["seg_next_p"] == 0  # r(1)=E, r(3)=T
    assert d["seg_from_n"] == 1 and d["seg_to_c"] == 8 - 1 - 2
    assert abs(d["seg_mass"] - (mz_yb[0] - PROTON)) < 1e-3


def test_empty_segments():
    site = np.zeros((2, 139), np.float32)
    x = build_internal_features(site, "PEK", np.zeros(0, int), np.zeros(0, int), np.zeros(0, int),
                                np.zeros(0, np.float32), 300.0, 2)
    assert x.shape == (0, len(internal_feature_names()))
