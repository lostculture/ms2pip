"""MS2 spectrum handling."""

from __future__ import annotations

import warnings
from typing import Annotated, Any

import numpy as np
from psm_utils import Peptidoform
from pydantic import BaseModel, BeforeValidator, ConfigDict, field_serializer, model_validator


def _coerce_peptidoform(v):
    if v is None or isinstance(v, Peptidoform):
        return v
    elif isinstance(v, str):
        return Peptidoform(v)
    raise ValueError("Peptidoform must be a string, a Peptidoform object, or None.")


_PeptidoformField = Annotated[Peptidoform | None, BeforeValidator(_coerce_peptidoform)]

_NEUTRON = 1.003355  # 13C - 12C mass difference
_HYDROGEN = 1.007825


class Spectrum(BaseModel):
    """
    MS2 spectrum.

    Parameters
    ----------
    mz
        Array of m/z values.
    intensity
        Array of intensity values.
    annotations
        Array of peak annotations.
    identifier
        Spectrum identifier.
    peptidoform
        Peptidoform.
    precursor_mz
        Precursor m/z.
    precursor_charge
        Precursor charge.
    retention_time
        Retention time.
    mass_tolerance
        Mass tolerance for spectrum annotation.
    mass_tolerance_unit
        Unit of mass tolerance for spectrum annotation.
    """

    mz: np.ndarray
    intensity: np.ndarray
    annotations: np.ndarray | None = None
    identifier: str | None = None
    peptidoform: _PeptidoformField = None
    precursor_mz: float | None = None
    precursor_charge: int | None = None
    retention_time: float | None = None
    mass_tolerance: float | None = None
    mass_tolerance_unit: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        return "{}.{}({})".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            f"identifier='{self.identifier}'",
        )

    @model_validator(mode="after")
    def check_array_lengths(self):
        if len(self.mz) != len(self.intensity):
            raise ValueError("Array lengths do not match.")
        if self.annotations is not None:
            if len(self.annotations) != len(self.intensity):
                raise ValueError("Array lengths do not match.")
        return self

    @field_serializer("mz", "intensity", "annotations")
    def _serialize_array(self, value: np.ndarray | None) -> list | None:
        return value.tolist() if value is not None else None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Spectrum):
            return NotImplemented
        return (
            np.array_equal(self.mz, other.mz)
            and np.array_equal(self.intensity, other.intensity)
            and (
                np.array_equal(self.annotations, other.annotations)
                if self.annotations is not None and other.annotations is not None
                else self.annotations is other.annotations
            )
            and self.identifier == other.identifier
            and self.peptidoform == other.peptidoform
            and self.precursor_mz == other.precursor_mz
            and self.precursor_charge == other.precursor_charge
            and self.retention_time == other.retention_time
            and self.mass_tolerance == other.mass_tolerance
            and self.mass_tolerance_unit == other.mass_tolerance_unit
        )

    @property
    def tic(self):
        """Total ion current."""
        return np.sum(self.intensity)

    def remove_reporter_ions(self, label_type=None) -> None:
        """Set the intensity of reporter ions to 0."""
        # TODO: Consider using the exact m/z values instead of a range.
        if label_type == "iTRAQ":
            for i, mz in enumerate(self.mz):
                if (mz >= 113) & (mz <= 118):
                    self.intensity[i] = 0

        # TMT6plex: 126.1277, 127.1311, 128.1344, 129.1378, 130.1411, 131.1382
        elif label_type == "TMT":
            for i, mz in enumerate(self.mz):
                if (mz >= 125) & (mz <= 132):
                    self.intensity[i] = 0

    def remove_precursor(self, tolerance=0.02) -> None:
        """Set the intensity of the precursor peak to 0."""
        if not self.precursor_mz:
            raise ValueError("Precursor m/z must be set.")
        for i, mz in enumerate(self.mz):
            if (mz >= self.precursor_mz - tolerance) & (mz <= self.precursor_mz + tolerance):
                self.intensity[i] = 0

    def remove_charge_reduced_precursor(self, tolerance=0.02, tolerance_mode="Da") -> None:
        """
        Set the intensity of the precursor and its charge-reduced species to 0.

        Electron-based fragmentation leaves much of the signal in the intact precursor
        and in charge-reduced forms ([M+zH]^(k)+· for k < z). Each is removed with
        isotopes 0-3 and hydrogen shifts of -2 to +2. Without a precursor charge, only
        the precursor m/z itself is removed.
        """
        if not self.precursor_mz:
            raise ValueError("Precursor m/z must be set.")
        if not self.precursor_charge:
            targets = np.array([self.precursor_mz])
        else:
            charge = self.precursor_charge
            mh = self.precursor_mz * charge  # [M+zH] mass, as seen at charge 1
            shifts = (
                np.arange(4)[:, None] * _NEUTRON + np.arange(-2, 3)[None, :] * _HYDROGEN
            ).ravel()
            targets = np.concatenate([(mh + shifts) / k for k in range(1, charge + 1)])
        if tolerance_mode.lower() == "ppm":
            window = targets * tolerance * 1e-6
        else:
            window = np.full(targets.shape, tolerance)
        hit = np.any(np.abs(self.mz[:, None] - targets[None, :]) <= window[None, :], axis=1)
        self.intensity = np.where(hit, 0, self.intensity)

    def tic_norm(self) -> None:
        """Normalize spectrum to total ion current."""
        self.intensity = self.intensity / self.tic

    def log2_transform(self) -> None:
        """Log2-transform spectrum."""
        self.intensity = np.log2(self.intensity + 0.001)

    def inverse_log2_transform(self) -> None:
        """Undo log2 transformation of intensities (inverse of :meth:`log2_transform`)."""
        self.intensity = (2**self.intensity) - 0.001

    def clip_intensity(self, min_intensity=0.0) -> None:
        """Clip intensity values."""
        self.intensity = np.clip(self.intensity, min_intensity, None)

    def to_spectrum_utils(self):
        """
        Convert to spectrum_utils.spectrum.MsmsSpectrum.

        Notes
        -----
        - Requires spectrum_utils to be installed.
        - If the ``precursor_mz`` or ``precursor_charge`` attributes are not set, the theoretical
          m/z and precursor charge of the ``peptidoform`` attribute are used, if present.
          Otherwise, ``ValueError`` is raised.

        """
        try:
            import spectrum_utils.spectrum as sus
        except ImportError as e:
            raise ImportError("Optional dependency spectrum_utils not installed.") from e

        if self.precursor_charge:
            precursor_charge = self.precursor_charge
        else:
            if not self.peptidoform:
                raise ValueError("`precursor_charge` or `peptidoform` must be set.")
            precursor_charge = self.peptidoform.precursor_charge
            if precursor_charge is None:
                raise ValueError("Peptidoform charge state is not set.")

        if self.precursor_mz:
            precursor_mz_float = float(self.precursor_mz)
        else:
            if not self.peptidoform:
                raise ValueError("`precursor_mz` or `peptidoform` must be set.")
            elif not self.peptidoform.theoretical_mz:
                raise ValueError(
                    "Peptidoform theoretical m/z could not be calculated; ensure the charge state "
                    " is set."
                )
            else:
                warnings.warn("precursor_mz not set, using theoretical precursor m/z.")
                precursor_mz_float = float(self.peptidoform.theoretical_mz)

        spectrum = sus.MsmsSpectrum(
            identifier=self.identifier if self.identifier else "spectrum",
            precursor_mz=precursor_mz_float,
            precursor_charge=precursor_charge,
            mz=self.mz,
            intensity=self.intensity,
            retention_time=self.retention_time if self.retention_time is not None else 0.0,
        )
        if (
            self.peptidoform
            and self.mass_tolerance is not None
            and self.mass_tolerance_unit is not None
        ):
            spectrum.annotate_proforma(
                str(self.peptidoform),
                self.mass_tolerance,
                self.mass_tolerance_unit,
            )
        return spectrum


class ObservedSpectrum(Spectrum):
    """Observed MS2 spectrum."""

    pass


class PredictedSpectrum(Spectrum):
    """Predicted MS2 spectrum."""

    mass_tolerance: float | None = 0.001
    mass_tolerance_unit: str | None = "Da"
