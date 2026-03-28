from __future__ import annotations

from typing import Optional


RAW_TO_CANONICAL = {
    "Broken soybeans": "Broken",
    "Immature soybeans": "Immature",
    "Intact soybeans": "Intact",
    "Skin-damaged soybeans": "Skin Damaged",
    "Spotted soybeans": "Spotted",
}

CANONICAL_LABELS = [
    "Broken",
    "Immature",
    "Intact",
    "Skin Damaged",
    "Spotted",
]

RAW_LABEL_ORDER = [
    "Broken soybeans",
    "Immature soybeans",
    "Intact soybeans",
    "Skin-damaged soybeans",
    "Spotted soybeans",
]

NORMALIZED_LOOKUP = {
    "broken": "Broken",
    "immature": "Immature",
    "intact": "Intact",
    "skin damaged": "Skin Damaged",
    "skin-damaged": "Skin Damaged",
    "spotted": "Spotted",
}


def canonicalize_dataset_label(raw_label: str) -> str:
    """Converte um rotulo bruto do dataset para o padrao canonico.

    Parametros:
        raw_label: Nome bruto vindo do manifesto de folds.

    Retorno:
        str: Rotulo canonico equivalente.
    """
    if raw_label not in RAW_TO_CANONICAL:
        raise ValueError(f"Classe inesperada no manifesto: {raw_label}")
    return RAW_TO_CANONICAL[raw_label]


def normalize_predicted_label(raw_text: str) -> Optional[str]:
    """Normaliza a saida textual do modelo para um rotulo canonico.

    Parametros:
        raw_text: Texto bruto retornado pelo modelo.

    Retorno:
        Optional[str]: Rotulo canonico reconhecido ou None.
    """
    cleaned = (raw_text or "").strip().strip('"').strip("'")
    if not cleaned:
        return None

    lower_cleaned = " ".join(cleaned.replace("\n", " ").split()).lower()
    if lower_cleaned in NORMALIZED_LOOKUP:
        return NORMALIZED_LOOKUP[lower_cleaned]

    for candidate, canonical in NORMALIZED_LOOKUP.items():
        if candidate in lower_cleaned:
            return canonical

    return None


def label_to_index(label: str) -> int:
    """Converte um rotulo canonico para indice numerico.

    Parametros:
        label: Rotulo canonico da classe.

    Retorno:
        int: Posicao do rotulo na ordem canonica global.
    """
    return CANONICAL_LABELS.index(label)
