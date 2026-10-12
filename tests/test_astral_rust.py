"""ms2rescore-rs kernels used by the Astral model: astral_hcd terminal series with named losses."""

import numpy as np
from ms2rescore_rs import MS2Spectrum, Precursor, annotate_ms2_spectra, ms2pip_compute_theoretical_mz, \
    ms2pip_extract_targets

PROTON, H2O, NH3, CO = 1.007276466812, 18.010564684, 17.026549101, 27.994914619570
P, E, T, I, D, K = 97.052764, 129.042593, 101.047679, 113.084064, 115.026943, 128.094963
KEYS = ["y", "b", "a", "y2", "b2", "y-h2o", "b-h2o", "y-nh3", "b-nh3", "a-nh3", "a-h2o"]


def test_theoretical_mz_loss_keys():
    mz = ms2pip_compute_theoretical_mz(["PEPTIDEK/2"], KEYS, "astral_hcd", "monoisotopic")[0]
    assert set(mz) == set(KEYS)
    assert all(len(v) == 7 for v in mz.values())
    b2 = P + E + PROTON
    np.testing.assert_allclose(mz["b"][1], b2, atol=1e-3)
    np.testing.assert_allclose(mz["b-h2o"][1], b2 - H2O, atol=1e-3)
    np.testing.assert_allclose(mz["a-nh3"][1], b2 - CO - NH3, atol=1e-3)
    y1 = K + H2O + PROTON
    np.testing.assert_allclose(mz["y"][0], y1, atol=1e-3)
    np.testing.assert_allclose(mz["y-nh3"][0], y1 - NH3, atol=1e-3)


def test_extract_targets_loss_keys():
    b2 = P + E + PROTON
    peaks = np.array(sorted([b2 - H2O, b2, 500.0]), dtype=np.float32)
    spectrum = MS2Spectrum(identifier="s", mz=peaks.tolist(), intensity=[1.0, 2.0, 3.0],
                           precursor=Precursor(mz=465.7, rt=0.0, im=0.0, charge=2))
    ann = annotate_ms2_spectra([spectrum], ["PEPTIDEK/2"], "astral_hcd", "monoisotopic", 10.0, "ppm")
    targets = ms2pip_extract_targets(ann, [np.array([-1.0, -2.0, -3.0], dtype=np.float32)], KEYS, [8])[0]
    assert targets["b-h2o"][1] == -1.0
    assert targets["b"][1] == -2.0
    assert targets["y"][0] < -9.9  # floor
