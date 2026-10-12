"""Internal fragment ions (yb, ya): pair features for the Astral model.

A segment covers residues r(s)..r(e) (0-based, 1 <= s < e <= n-2). Its two cleavage sites are feature
rows s-1 (between r(s-1) and r(s)) and e (between r(e) and r(e+1)) of ms2rescore-rs's per-site matrix.
The segment list, its order and the isobaric collapsing come from ms2rescore-rs
(`ms2pip_internal_fragments`); `oracle_segments` re-derives them in Python for tests only.
"""

from __future__ import annotations

import numpy as np

from ms2pip._utils.feature_names import get_feature_names

PROTON = 1.007276466812
NUM_SITE_FEATURES = 139
DESCRIPTOR_NAMES = [
    "seg_len", "seg_mass", "seg_n_k", "seg_n_r", "seg_n_h", "seg_n_p", "seg_n_d", "seg_n_e",
    "seg_start_p", "seg_next_p", "seg_from_n", "seg_to_c", "seg_mass_frac", "seg_n_isobaric",
    "pep_len", "pep_charge",
]


def internal_feature_names() -> list[str]:
    site = get_feature_names()
    return [f"n_{f}" for f in site] + [f"c_{f}" for f in site] + DESCRIPTOR_NAMES


def descriptor_columns() -> list[str]:
    return list(DESCRIPTOR_NAMES)


def build_internal_features(site_matrix: np.ndarray, sequence: str, s: np.ndarray, e: np.ndarray,
                            n_isobaric: np.ndarray, mz_yb: np.ndarray, precursor_mass: float,
                            charge: int) -> np.ndarray:
    """Feature rows (k, 2*139 + 16) for the k segments of one peptide."""
    s = np.asarray(s, dtype=np.int64)
    e = np.asarray(e, dtype=np.int64)
    k, n = len(s), len(sequence)
    out = np.empty((k, 2 * NUM_SITE_FEATURES + len(DESCRIPTOR_NAMES)), dtype=np.float32)
    if k == 0:
        return out
    out[:, :NUM_SITE_FEATURES] = site_matrix[s - 1]
    out[:, NUM_SITE_FEATURES:2 * NUM_SITE_FEATURES] = site_matrix[e]
    seq = np.frombuffer(sequence.encode("ascii"), dtype="S1")
    counts = {aa: np.cumsum(np.concatenate([[0], seq == aa.encode()])) for aa in "KRHPDE"}
    mass = np.asarray(mz_yb, dtype=np.float64) - PROTON
    desc = np.column_stack([
        e - s + 1,
        mass,
        *[counts[aa][e + 1] - counts[aa][s] for aa in "KRHPDE"],
        seq[s] == b"P",
        seq[e + 1] == b"P",
        s,
        n - 1 - e,
        # seg_mass_frac. precursor_mass is the PSM peptidoform's theoretical_mass: training PSMs
        # carry 4-decimal mass-shift ProForma (e.g. [+15.9949]), inference may carry the exact
        # modification mass; the difference is ~1e-8 relative, negligible.
        mass / precursor_mass,
        n_isobaric,
        np.full(k, n),
        np.full(k, charge),
    ])
    out[:, 2 * NUM_SITE_FEATURES:] = desc
    return out


def oracle_segments(residue_masses: list[float], max_mass: float) -> list[tuple[int, int, int]]:
    """(s, e, n_isobaric) by the spec's rule, from per-residue masses (test oracle for the Rust list)."""
    n = len(residue_masses)
    groups: dict[tuple, int] = {}
    out: list[list[int]] = []
    for s in range(1, n - 2):
        for e in range(s + 1, n - 1):
            mass = sum(residue_masses[s:e + 1])
            if mass >= max_mass:
                break
            key = tuple(sorted(round(m * 1e4) for m in residue_masses[s:e + 1]))
            if key in groups:
                out[groups[key]][2] += 1
            else:
                groups[key] = len(out)
                out.append([s, e, 1])
    return [tuple(x) for x in out]
