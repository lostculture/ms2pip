"""Definition and handling of MS²PIP results."""

from __future__ import annotations

import csv
from logging import getLogger
from pathlib import Path
import numpy as np
from psm_utils import PSM
from pydantic import BaseModel, ConfigDict

from ms2pip._utils.ion_types import annotation_label, internal_label
from ms2pip.correlation import pearson
from ms2pip.spectrum import ObservedSpectrum, PredictedSpectrum

logger = getLogger(__name__)


class ProcessingResult(BaseModel):
    """
    Result of processing a single PSM.

    Parameters
    ----------
    psm_index
        Index of the PSM in the input list.
    psm
        The PSM object.
    theoretical_mz
        Dict mapping ion type to theoretical m/z array.
    predicted_intensity
        Dict mapping ion type to predicted intensity array.
    observed_intensity
        Dict mapping ion type to observed intensity array.
    correlation
        Pearson correlation between predicted and observed intensities.
    correlation_internal
        Pearson correlation over internal fragments (models with internal ion types).
    feature_vectors
        Feature vectors for model training.
    internal
        Internal fragments (models with internal_ion_types): s, e, n_isobaric, mz, observed,
        predicted, features.
    """

    psm_index: int
    psm: PSM
    theoretical_mz: dict[str, np.ndarray] | None = None
    predicted_intensity: dict[str, np.ndarray] | None = None
    observed_intensity: dict[str, np.ndarray] | None = None
    correlation: float | None = None
    correlation_internal: float | None = None
    feature_vectors: np.ndarray | None = None
    internal: dict | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def as_spectra(self) -> tuple[PredictedSpectrum | None, ObservedSpectrum | None]:
        """Convert result to predicted and observed spectra."""
        if not self.theoretical_mz:
            raise ValueError("Theoretical m/z values required to convert to spectra.")
        # The dicts can list ion types in different orders (ms2rescore-rs returns
        # HashMaps), so always look them up by name in one fixed order.
        ion_types = list(self.theoretical_mz)
        mz = np.concatenate([self.theoretical_mz[it] for it in ion_types])
        annotations = np.concatenate(
            [
                [annotation_label(it, i + 1) for i in range(len(self.theoretical_mz[it]))]
                for it in ion_types
            ]
        )
        pred_parts = [self.predicted_intensity[it] for it in ion_types] if self.predicted_intensity else None
        obs_parts = [self.observed_intensity[it] for it in ion_types] if self.observed_intensity else None
        if self.internal:
            block = self.internal
            for key in block["mz"]:
                mz = np.concatenate([mz, block["mz"][key]])
                annotations = np.concatenate(
                    [
                        annotations,
                        [internal_label(key, s, e) for s, e in zip(block["s"], block["e"])],
                    ]
                )
                if pred_parts is not None:
                    pred_parts.append(block["predicted"][key])
                if obs_parts is not None:
                    obs_parts.append(block["observed"][key])
        peak_order = np.argsort(mz)

        if self.predicted_intensity:
            pred_int = np.concatenate(pred_parts)
            predicted = PredictedSpectrum(
                mz=mz[peak_order],
                intensity=pred_int[peak_order],
                annotations=annotations[peak_order],
                peptidoform=self.psm.peptidoform if self.psm else None,
                precursor_charge=self.psm.peptidoform.precursor_charge if self.psm else None,
            )
            predicted.inverse_log2_transform()
        else:
            predicted = None

        if self.observed_intensity:
            obs_int = np.concatenate(obs_parts)
            observed = ObservedSpectrum(
                mz=mz[peak_order],
                intensity=obs_int[peak_order],
                annotations=annotations[peak_order],
                peptidoform=self.psm.peptidoform if self.psm else None,
                precursor_charge=self.psm.peptidoform.precursor_charge if self.psm else None,
            )
            observed.inverse_log2_transform()
        else:
            observed = None

        return predicted, observed

    def plot_spectra(self):
        """
        Plot predicted and observed spectra.

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        -----
        Requires optional dependency ``spectrum_utils`` to be installed.

        """
        try:
            import spectrum_utils.plot as sup
        except ImportError as e:
            raise ImportError("Optional dependency spectrum_utils not installed.") from e

        predicted, observed = (
            spec.to_spectrum_utils() if spec else None for spec in self.as_spectra()
        )
        if predicted and observed:
            ax = sup.mirror(observed, predicted)
            ax.set_title(
                f"Observed (top) and predicted (bottom) spectra for {self.psm.peptidoform}"
            )
        elif predicted:
            ax = sup.spectrum(predicted)
            ax.set_title(f"Predicted spectrum for {self.psm.peptidoform}")
        elif observed:
            ax = sup.spectrum(observed)
            ax.set_title(f"Observed spectrum for {self.psm.peptidoform}")
        else:
            raise ValueError("No spectra to plot.")
        return ax


def mask_mz(
    values: dict[str, np.ndarray],
    mz: dict[str, np.ndarray],
    min_mz: float,
    max_mz: float | None = None,
) -> dict[str, np.ndarray]:
    """Keep positions whose theoretical m/z is generated (> 0), >= min_mz and (if set) <= max_mz."""
    out = {}
    for k, v in values.items():
        keep = (mz[k] > 0) & (mz[k] >= min_mz)
        if max_mz is not None:
            keep &= mz[k] <= max_mz
        out[k] = v[keep]
    return out


def mask_low_mz(
    values: dict[str, np.ndarray], mz: dict[str, np.ndarray], min_mz: float
) -> dict[str, np.ndarray]:
    """Alias of ``mask_mz`` without an upper bound."""
    return mask_mz(values, mz, min_mz)


def calculate_correlations(results: list[ProcessingResult], model: str | None = None) -> None:
    """
    Calculate and add Pearson correlations to list of results.

    For models with ``min_fragment_mz`` / ``max_fragment_mz``, positions outside that range (or not generated) are excluded.
    Internal fragments get their own ``correlation_internal``.
    """
    # TODO: Consider nan values? https://github.com/CompOmics/ms2pip/pull/214/changes#diff-5f77421a48cf8f17c5b83ed031897ce8076e2f52c0028aa6f4294a34ba3b3305R115-R123
    from ms2pip.constants import MODELS

    params = (MODELS.get(model) or {}) if model else {}
    min_mz, max_mz = params.get("min_fragment_mz"), params.get("max_fragment_mz")
    for result in results:
        if result.predicted_intensity is None or result.observed_intensity is None:
            continue
        # Pair ion types by name: the observed dict's order can differ (Rust HashMap).
        ion_types = list(result.predicted_intensity)
        pred, obs = result.predicted_intensity, result.observed_intensity
        if min_mz and result.theoretical_mz:
            pred = mask_mz(pred, result.theoretical_mz, min_mz, max_mz)
            obs = mask_mz({k: obs[k] for k in ion_types}, result.theoretical_mz, min_mz, max_mz)
        result.correlation = pearson(
            np.concatenate([pred[it] for it in ion_types]),
            np.concatenate([obs[it] for it in ion_types]),
        )
        block = result.internal
        if block and block.get("predicted") is not None and block.get("observed") is not None:
            keys = list(block["predicted"])
            ip = mask_mz(block["predicted"], block["mz"], min_mz or 0.0, max_mz)
            io = mask_mz({k: block["observed"][k] for k in keys}, block["mz"], min_mz or 0.0, max_mz)
            result.correlation_internal = pearson(
                np.concatenate([ip[k] for k in keys]), np.concatenate([io[k] for k in keys])
            )


def write_correlations(results: list[ProcessingResult], output_file: str | Path) -> None:
    """Write correlations to CSV file."""
    with open(output_file, "wt") as f:
        fieldnames = ["psm_index", "correlation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({"psm_index": result.psm_index, "correlation": result.correlation})
