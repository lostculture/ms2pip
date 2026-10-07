import numpy as np

from ms2pip.correlation import ms2pip_pearson, spectral_angle


def test_ms2pip_pearson():
    true = np.array([100.0, 200.0, 50.0, 300.0])
    pred = np.array([90.0, 210.0, 40.0, 310.0])

    corr = ms2pip_pearson(true, pred)

    assert isinstance(corr, float)
    assert 0.9 < corr <= 1.0


def test_ms2pip_pearson_identical():
    arr = np.array([100.0, 200.0, 50.0, 300.0])
    corr = ms2pip_pearson(arr, arr)

    assert abs(corr - 1.0) < 1e-6


def test_ms2pip_pearson_anticorrelated():
    true = np.array([100.0, 200.0, 300.0])
    pred = np.array([300.0, 200.0, 100.0])

    corr = ms2pip_pearson(true, pred)

    assert corr < 0


def test_spectral_angle():
    true = np.array([100.0, 200.0, 50.0, 300.0])
    pred = np.array([90.0, 210.0, 40.0, 310.0])

    sa = spectral_angle(true, pred)

    assert isinstance(sa, float)
    assert 0.0 < sa <= 1.0


def test_spectral_angle_identical():
    arr = np.array([100.0, 200.0, 50.0, 300.0])
    sa = spectral_angle(arr, arr)

    assert abs(sa - 1.0) < 1e-6


def test_spectral_angle_orthogonal():
    true = np.array([1.0, 0.0])
    pred = np.array([0.0, 1.0])

    sa = spectral_angle(true, pred)

    assert sa < 0.5


def _result_with_shuffled_keys():
    """Result whose dicts list the same ion types in different orders.

    ms2rescore-rs returns observed intensities and theoretical m/z as Rust
    HashMaps, so their key order varies from PSM to PSM.
    """
    from psm_utils import PSM

    from ms2pip.result import ProcessingResult

    b = np.array([-1.0, -2.0, -3.0])
    y = np.array([-9.0, -5.0, -7.0])
    return ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform="PEPK/2", spectrum_id=0),
        theoretical_mz={"y": np.array([147.1, 244.2, 341.2]), "b": np.array([98.1, 227.1, 324.2])},
        predicted_intensity={"b": b, "y": y},
        observed_intensity={"y": y, "b": b},
    )


def test_calculate_correlations_aligns_ion_types_by_name():
    from ms2pip.result import calculate_correlations

    result = _result_with_shuffled_keys()
    calculate_correlations([result])
    assert abs(result.correlation - 1.0) < 1e-6


def test_as_spectra_pairs_mz_with_intensity_by_ion_type():
    predicted, observed = _result_with_shuffled_keys().as_spectra()
    by_mz = dict(zip(np.round(predicted.mz, 1), np.round(np.log2(predicted.intensity + 0.001), 1)))
    assert by_mz[98.1] == -1.0  # b1
    assert by_mz[147.1] == -9.0  # y1
    np.testing.assert_allclose(predicted.intensity, observed.intensity)
