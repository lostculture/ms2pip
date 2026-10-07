"""Ion-type keys, matching the grammar used by ms2rescore-rs.

``<series>[<signed variant>][<charge>]``. Variant 0 keeps the legacy form
(``y``, ``y2`` = doubly charged y). Non-zero hydrogen-shift variants separate
the charge with an underscore: ``z+1``, ``z+1_2`` (doubly charged z+1), ``c-1``.
"""

from __future__ import annotations

import re

C_TERMINAL_SERIES = frozenset("xyz")

_ION_TYPE = re.compile(
    r"^(?P<series>[abcxyz])(?:(?P<variant>[+-][1-9]\d*)(?:_(?P<vcharge>[1-9]\d*))?|(?P<charge>[1-9]\d*))?$"
)


def parse_ion_type(ion_type: str) -> tuple[str, int, int]:
    """Parse an ion-type key into (series, variant, charge)."""
    match = _ION_TYPE.match(ion_type.lower())
    if not match:
        raise ValueError(f"Unsupported ion type: {ion_type}")
    variant = int(match["variant"]) if match["variant"] else 0
    charge = int(match["vcharge"] or match["charge"] or 1)
    return match["series"], variant, charge


def is_c_terminal(ion_type: str) -> bool:
    """True for x/y/z series, whose ions are numbered from the C-terminus."""
    return parse_ion_type(ion_type)[0] in C_TERMINAL_SERIES


def annotation_label(ion_type: str, position: int) -> str:
    """Peak label for ion ``position`` (1-based) of ``ion_type``.

    Legacy keys keep their existing form (``y2`` -> ``y23``); variant ions are
    written ``z3+1``, with ``^charge`` when the charge is above 1 (``z3+1^2``).
    """
    series, variant, charge = parse_ion_type(ion_type)
    if variant == 0:
        return f"{ion_type}{position}"
    return f"{series}{position}{variant:+d}" + (f"^{charge}" if charge > 1 else "")
