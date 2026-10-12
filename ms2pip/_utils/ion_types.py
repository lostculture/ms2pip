"""Ion-type keys, matching the grammar used by ms2rescore-rs.

``<series>[<signed variant>][<charge>][-<loss>]``. Variant 0 keeps the legacy form
(``y``, ``y2`` = doubly charged y). Non-zero hydrogen-shift variants separate
the charge with an underscore: ``z+1``, ``z+1_2`` (doubly charged z+1), ``c-1``.
A neutral loss is appended after a dash to variant-0 keys only: ``b-h2o``,
``y2-nh3``; ``z+1-h2o`` is rejected.

Internal fragment keys (``yb``, ``ya``) are indexed by segment ``[s-e]`` rather
than by cleavage site; see ``is_internal`` and ``internal_label``.
"""

from __future__ import annotations

import re

C_TERMINAL_SERIES = frozenset("xyz")
INTERNAL_TYPES = frozenset({"yb", "ya"})
LOSSES = {"h2o": "H2O", "nh3": "NH3"}

_ION_TYPE = re.compile(
    r"^(?P<series>[abcxyz])(?:(?P<variant>[+-][1-9]\d*)(?:_(?P<vcharge>[1-9]\d*))?|(?P<charge>[1-9]\d*))?"
    r"(?:-(?P<loss>h2o|nh3))?$"
)


def is_internal(ion_type: str) -> bool:
    """True for internal fragment keys (yb, ya), which are indexed by segment, not by cleavage site."""
    return ion_type.lower() in INTERNAL_TYPES


def parse_ion_type(ion_type: str) -> tuple[str, int, int, str]:
    """Parse a terminal ion-type key into (series, variant, charge, loss)."""
    match = _ION_TYPE.match(ion_type.lower())
    if not match or (match["loss"] and match["variant"]):
        raise ValueError(f"Unsupported ion type: {ion_type}")
    variant = int(match["variant"]) if match["variant"] else 0
    charge = int(match["vcharge"] or match["charge"] or 1)
    return match["series"], variant, charge, match["loss"] or ""


def is_c_terminal(ion_type: str) -> bool:
    """True for x/y/z series, whose ions are numbered from the C-terminus. False for internal keys."""
    if is_internal(ion_type):
        return False
    return parse_ion_type(ion_type)[0] in C_TERMINAL_SERIES


def annotation_label(ion_type: str, position: int) -> str:
    """Peak label for ion ``position`` (1-based) of a terminal ``ion_type``.

    Legacy keys keep their existing form (``y2`` -> ``y23``); variant ions are written ``z3+1``, with
    ``^charge`` above 1 (``z3+1^2``); losses are appended (``b3-H2O``, ``y23-NH3``).
    """
    series, variant, charge, loss = parse_ion_type(ion_type)
    suffix = f"-{LOSSES[loss]}" if loss else ""
    if variant == 0:
        base = ion_type.lower().removesuffix(f"-{loss}") if loss else ion_type
        return f"{base}{position}{suffix}"
    return f"{series}{position}{variant:+d}" + (f"^{charge}" if charge > 1 else "")


def internal_label(ion_type: str, s: int, e: int) -> str:
    """Peak label for an internal fragment covering residues s..e (0-based): ``yb[3-5]``."""
    return f"{ion_type.lower()}[{s}-{e}]"
