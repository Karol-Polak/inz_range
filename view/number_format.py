"""Polish number formatting for user-visible text (decimal comma).

Display only - stored values and CSV export keep the dot separator.
The single implementation lives in services.units.format_number.
"""

from services.units import format_number as fmt_num  # noqa: F401  (re-export)


def fmt_general(value) -> str:
    """Shortest representation (like :g), e.g. 10 -> '10', 10.5 -> '10,5'."""
    return f"{float(value):g}".replace(".", ",")
