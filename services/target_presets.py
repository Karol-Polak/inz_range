"""
Katalog standardowych wymiarów tarcz (elementów odniesienia do kalibracji skali).

Czysta logika (bez PyQt). Wszystkie wymiary w milimetrach i NOMINALNE -
obowiązują tolerancje druku, a konkretna tarcza może się nieznacznie różnić.

Źródła danych (podane przy każdej tarczy w polu `source`):
  - tarcze zgodne z ISSF: Eemann Tech, zielsport.at, TargetLog,
  - tarcza papierowa 17 × 17 cm: opis sklepu (zakupywgodzine.pl),
  - IPSC: wymiary podane przez autora pracy (nieweryfikowane niezależnie;
    przepisy IPSC, dodatek B2).

`needs_check=True` oznacza dane z pojedynczego źródła albo nieweryfikowane
niezależnie - interfejs pokazuje wtedy ostrzeżenie, żeby sprawdzić wymiar
na własnej tarczy.

Celowo POMINIĘTO sylwetki policyjne oraz ISSF 25 m (szybkostrzelna) -
ich wymiarów nie udało się zweryfikować.

Wybór elementu zmienia wyłącznie wartość wejściową "znana odległość
rzeczywista (mm)"; obliczenie skali (mm / px) pozostaje bez zmian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ReferenceFeature:
    label: str
    size_mm: float
    note: str = ""


@dataclass(frozen=True)
class TargetPreset:
    id: str
    name: str
    features: Tuple[ReferenceFeature, ...]
    source: str
    confidence: str
    needs_check: bool
    round_target: bool


_PRESETS: Tuple[TargetPreset, ...] = (
    TargetPreset(
        id="issf_25m_precision",
        name="ISSF 25 m precyzyjna / 50 m pistolet",
        features=(
            ReferenceFeature("pierścień 1 (zewnętrzny)", 500.0),
            ReferenceFeature("czerń (pierścienie 7–10)", 200.0),
            ReferenceFeature("dziesiątka", 50.0),
            ReferenceFeature("arkusz", 550.0, note="550 × 550 mm"),
        ),
        source="tarcze zgodne z ISSF (Eemann Tech, zielsport.at, TargetLog)",
        confidence="potwierdzone w 3 źródłach",
        needs_check=False,
        round_target=True,
    ),
    TargetPreset(
        id="issf_10m_air_pistol",
        name="ISSF 10 m pistolet pneumatyczny",
        features=(
            ReferenceFeature("pierścień 1 (zewnętrzny)", 155.5),
            ReferenceFeature("czerń (pierścienie 7–10)", 59.5),
            ReferenceFeature("dziesiątka", 11.5),
            ReferenceFeature("karta (min.)", 170.0, note="170 × 170 mm"),
        ),
        source="zielsport.at (ISSF), TargetLog",
        confidence="potwierdzone w 2 źródłach",
        needs_check=False,
        round_target=True,
    ),
    TargetPreset(
        id="issf_10m_air_rifle",
        name="ISSF 10 m karabinek pneumatyczny",
        features=(
            ReferenceFeature("pierścień 1 (zewnętrzny)", 45.5),
            ReferenceFeature("czerń (pierścienie 4–9)", 30.5),
        ),
        source="TargetLog",
        confidence="pojedyncze źródło – sprawdź na swojej tarczy",
        needs_check=True,
        round_target=True,
    ),
    TargetPreset(
        id="issf_50m_rifle",
        name="ISSF 50 m karabin",
        features=(
            ReferenceFeature("pierścień 1 (zewnętrzny)", 154.4),
            ReferenceFeature("czerń (pierścienie 4–10)", 112.4),
            ReferenceFeature("dziesiątka", 10.4),
        ),
        source="TargetLog",
        confidence="pojedyncze źródło – sprawdź na swojej tarczy",
        needs_check=True,
        round_target=True,
    ),
    TargetPreset(
        id="paper_17x17",
        name="Tarcza papierowa 17 × 17 cm",
        features=(
            ReferenceFeature("bok arkusza", 170.0),
        ),
        source="opis sklepu (zakupywgodzine.pl)",
        confidence="wymiar arkusza; producenci mogą się różnić",
        needs_check=True,
        round_target=False,
    ),
    TargetPreset(
        id="ipsc",
        name="IPSC (tarcza kartonowa)",
        features=(
            ReferenceFeature("szerokość", 460.0),
            ReferenceFeature("wysokość", 580.0),
        ),
        source="podane przez autora (nieweryfikowane niezależnie)",
        confidence="sprawdź na swojej tarczy (przepisy IPSC, dodatek B2)",
        needs_check=True,
        round_target=False,
    ),
)


def all_presets() -> Tuple[TargetPreset, ...]:
    return _PRESETS


def get_preset(preset_id: str) -> Optional[TargetPreset]:
    for preset in _PRESETS:
        if preset.id == preset_id:
            return preset
    return None


def find_feature(preset_id: str, label: str) -> Optional[ReferenceFeature]:
    preset = get_preset(preset_id)
    if preset is None:
        return None
    for feature in preset.features:
        if feature.label == label:
            return feature
    return None


def _format_mm(size_mm: float) -> str:
    if float(size_mm).is_integer():
        return f"{int(size_mm)}"
    return f"{size_mm:.1f}".replace(".", ",")


def feature_display_text(feature: ReferenceFeature) -> str:
    """Np. 'czerń (pierścienie 7–10) – 200 mm', 'dziesiątka – 11,5 mm'."""
    return f"{feature.label} – {_format_mm(feature.size_mm)} mm"


def measurement_hint(preset: TargetPreset) -> Optional[str]:
    """
    Wskazówka pomiaru dla tarcz nieokrągłych (round_target=False); None dla
    okrągłych. Skala jest jedna (izotropowa) - pomiar w dowolnym kierunku
    daje tę samą skalę tylko przy zdjęciu zrobionym na wprost.
    """
    if preset.round_target:
        return None
    if len(preset.features) == 1:
        return (
            f"Zmierz {preset.features[0].label} przyciskiem „Zmierz na zdjęciu”. "
            "Arkusz powinien leżeć płasko, a zdjęcie być zrobione na wprost."
        )
    return (
        "Tarcza nie jest okrągła – zmierz szerokość lub wysokość przyciskiem "
        "„Zmierz na zdjęciu”. Jeśli obie dają wyraźnie różną skalę, zdjęcie "
        "jest zrobione pod kątem."
    )

