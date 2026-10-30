"""
Jednostki prezentacji długości: px, mm, MOA, mrad.

Czysta logika (bez PyQt), testowalna bez ekranu. Baza danych przechowuje
wartości w pikselach + skalę mm/px + odległość strzelania; przeliczenia
dzieją się wyłącznie w momencie wyświetlania.

Definicje kątowe:
  - 1 MOA = 1/60 stopnia = pi/10800 rad (MOA kątowe, NIE "shooter's MOA"
    1,047 cala na 100 jardów). Na 100 m: 1 MOA = 100000 mm * tan(pi/10800)
    = 29,0888 mm.
  - kąt = atan(L / D), L - długość w mm, D - odległość strzelania w mm.
    MOA = degrees(kąt) * 60, mrad = 1000 * kąt.

Używamy dokładnej postaci trygonometrycznej (atan). Względna różnica
względem przybliżenia małego kąta (L/D) wynosi około (L/D)^2 / 3, np.
~1,3e-6 dla 50 mm na 25 m, ale ~7,5e-5 dla 150 mm na 10 m - w obu
przypadkach dużo poniżej wyświetlanej dokładności (2 miejsca po przecinku).

Rozrzut maksymalny (ES) przeliczamy tym samym wzorem atan(ES / D).
Ściśle kąt, pod jakim widać odcinek ES, to 2*atan(ES / 2D); różnica jest
rzędu (ES/D)^2 / 4 względnie, czyli również dużo poniżej wyświetlanej
dokładności.

atan jest funkcją nieparzystą, więc wartości ze znakiem (średnie
odchylenie X/Y) zachowują znak.

Brak skali lub odległości => jednostka niedostępna (None) - niczego nie
zgadujemy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple


class LengthUnit(str, Enum):
    PX = "px"
    MM = "mm"
    MOA = "MOA"
    MRAD = "mrad"


class MissingInput(str, Enum):
    SCALE = "scale"
    DISTANCE = "distance"


@dataclass(frozen=True)
class UnitContext:
    """Dane potrzebne do przeliczeń jednej sesji. None lub <= 0 = brak."""

    scale_mm_per_px: Optional[float] = None
    distance_m: Optional[float] = None

    @property
    def has_scale(self) -> bool:
        return _positive(self.scale_mm_per_px)

    @property
    def has_distance(self) -> bool:
        return _positive(self.distance_m)


ALL_UNITS: Tuple[LengthUnit, ...] = (LengthUnit.PX, LengthUnit.MM, LengthUnit.MOA, LengthUnit.MRAD)

DECIMALS = {
    LengthUnit.PX: 1,
    LengthUnit.MM: 1,
    LengthUnit.MOA: 2,
    LengthUnit.MRAD: 2,
}

# Miary długości w słowniku statystyk (services/statistics.py) - przeliczane.
# Liczności (count, outliers_count) nie są przeliczane nigdy.
LENGTH_STATISTICS = (
    "accuracy_radius",
    "precision_radius",
    "precision_std",
    "precision_max_radius",
    "cep_50",
    "p25",
    "p75",
    "std_x",
    "std_y",
    "extreme_spread",
    "mean_x",
    "mean_y",
)

_MOA_PER_RAD = 180.0 * 60.0 / math.pi


def _positive(value: Optional[float]) -> bool:
    return value is not None and value > 0


# ---------------------------------------------------------------------------
# Dostępność jednostek
# ---------------------------------------------------------------------------

def missing_inputs(unit: LengthUnit, ctx: UnitContext) -> Tuple[MissingInput, ...]:
    """Czego brakuje, żeby pokazać wartości w tej jednostce. () = dostępna."""
    unit = LengthUnit(unit)
    missing = []
    if unit in (LengthUnit.MM, LengthUnit.MOA, LengthUnit.MRAD) and not ctx.has_scale:
        missing.append(MissingInput.SCALE)
    if unit in (LengthUnit.MOA, LengthUnit.MRAD) and not ctx.has_distance:
        missing.append(MissingInput.DISTANCE)
    return tuple(missing)


def is_available(unit: LengthUnit, ctx: UnitContext) -> bool:
    return not missing_inputs(unit, ctx)


def available_units(ctx: UnitContext) -> List[LengthUnit]:
    return [unit for unit in ALL_UNITS if is_available(unit, ctx)]


def default_unit(ctx: UnitContext) -> LengthUnit:
    """mm, gdy skala jest znana; w przeciwnym razie px."""
    return LengthUnit.MM if ctx.has_scale else LengthUnit.PX


def missing_reason_text(unit: LengthUnit, ctx: UnitContext) -> Optional[str]:
    """Polski komunikat, dlaczego jednostka jest niedostępna (None = dostępna)."""
    unit = LengthUnit(unit)
    missing = set(missing_inputs(unit, ctx))
    if not missing:
        return None

    if unit is LengthUnit.MM:
        return "Ustaw skalę w Ustawieniach, aby zobaczyć wyniki w mm."

    if missing == {MissingInput.SCALE, MissingInput.DISTANCE}:
        return "MOA i mrad wymagają skali i odległości strzelania."
    if MissingInput.SCALE in missing:
        return "MOA i mrad wymagają skali (Ustawienia → Kalibracja skali)."
    return (
        "MOA i mrad wymagają odległości strzelania "
        "(pole „Odległość strzelania” w Zapisie sesji)."
    )


def common_unit(
    requested: LengthUnit, contexts: Sequence[UnitContext]
) -> Tuple[LengthUnit, Optional[str]]:
    """
    Wspólna jednostka dla kilku sesji, z których każda jest przeliczana
    WŁASNĄ skalą i odległością. Jeśli którejś sesji brakuje danych dla
    żądanej jednostki, wraca najlepsza wspólna: mm (gdy wszystkie mają
    skalę), inaczej px - razem z polskim ostrzeżeniem.
    """

    requested = LengthUnit(requested)
    contexts = list(contexts)
    total = len(contexts)

    lacking = [ctx for ctx in contexts if not is_available(requested, ctx)]
    if not lacking:
        return requested, None

    fallback = LengthUnit.MM if all(ctx.has_scale for ctx in contexts) else LengthUnit.PX
    if fallback is requested:  # px jest zawsze dostępne; tu nie powinno się zdarzyć
        return requested, None

    no_scale = sum(1 for ctx in lacking if not ctx.has_scale)
    no_distance = sum(
        1 for ctx in lacking
        if ctx.has_scale and requested in (LengthUnit.MOA, LengthUnit.MRAD) and not ctx.has_distance
    )

    reasons = []
    if no_scale:
        reasons.append(f"{no_scale} z {total} sesji nie ma zapisanej skali")
    if no_distance:
        reasons.append(
            f"{no_distance} z {total} sesji nie ma zapisanej odległości strzelania"
        )

    warning = (
        f"Uwaga: {' i '.join(reasons)} – wyniki pokazane w {fallback.value} "
        f"zamiast {requested.value}."
    )
    return fallback, warning


# ---------------------------------------------------------------------------
# Przeliczenia
# ---------------------------------------------------------------------------

def mm_to_moa(length_mm: float, distance_m: float) -> float:
    return math.atan(length_mm / (distance_m * 1000.0)) * _MOA_PER_RAD


def mm_to_mrad(length_mm: float, distance_m: float) -> float:
    return math.atan(length_mm / (distance_m * 1000.0)) * 1000.0


def moa_to_mm(moa: float, distance_m: float) -> float:
    return math.tan(moa / _MOA_PER_RAD) * distance_m * 1000.0


def mrad_to_mm(mrad: float, distance_m: float) -> float:
    return math.tan(mrad / 1000.0) * distance_m * 1000.0


def convert_length(value_px: float, unit: LengthUnit, ctx: UnitContext) -> Optional[float]:
    """Długość w px -> wybrana jednostka. None, gdy brakuje skali/odległości."""
    unit = LengthUnit(unit)
    if not is_available(unit, ctx):
        return None

    value_px = float(value_px)
    if unit is LengthUnit.PX:
        return value_px

    length_mm = value_px * ctx.scale_mm_per_px
    if unit is LengthUnit.MM:
        return length_mm
    if unit is LengthUnit.MOA:
        return mm_to_moa(length_mm, ctx.distance_m)
    return mm_to_mrad(length_mm, ctx.distance_m)


def convert_statistics(statistics: dict, unit: LengthUnit, ctx: UnitContext) -> Optional[dict]:
    """
    Kopia słownika statystyk z miarami długości (LENGTH_STATISTICS)
    przeliczonymi do jednostki; liczności bez zmian. None, gdy niedostępna.
    """
    if not is_available(unit, ctx):
        return None
    converted = dict(statistics)
    for key in LENGTH_STATISTICS:
        if key in converted and converted[key] is not None:
            converted[key] = convert_length(converted[key], unit, ctx)
    return converted


def convert_many(values_px: Iterable[float], unit: LengthUnit, ctx: UnitContext) -> Optional[List[float]]:
    if not is_available(unit, ctx):
        return None
    return [convert_length(value, unit, ctx) for value in values_px]


# ---------------------------------------------------------------------------
# Formatowanie (polski przecinek dziesiętny)
# ---------------------------------------------------------------------------

def format_number(value: float, decimals: int = 1, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{decimals}f}".replace(".", ",")


def format_length(value: float, unit: LengthUnit, signed: bool = False) -> str:
    unit = LengthUnit(unit)
    return f"{format_number(value, DECIMALS[unit], signed)} {unit.value}"
