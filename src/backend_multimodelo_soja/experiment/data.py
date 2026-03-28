from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..tasks.label_normalization import (
    CANONICAL_LABELS,
    RAW_LABEL_ORDER,
    RAW_TO_CANONICAL,
    canonicalize_dataset_label,
)


REQUIRED_MANIFEST_COLUMNS = ["fold", "subset", "path", "label_idx", "label_name"]


@dataclass
class SampleRecord:
    """Representa uma amostra individual lida do manifesto.

    Parametros:
        fold: Indice do fold.
        subset: Nome do subset.
        path: Caminho absoluto da imagem.
        label_idx: Indice numerico bruto.
        raw_label_name: Nome bruto da classe.
        canonical_label: Nome canonico da classe.
    """

    fold: int
    subset: str
    path: str
    label_idx: int
    raw_label_name: str
    canonical_label: str


class SoyManifestDataset(Dataset):
    """Dataset leve baseado apenas nos manifests externos."""

    def __init__(self, records: List[SampleRecord], max_image_size: int):
        """Inicializa o dataset do fold.

        Parametros:
            records: Lista de registros ja validados.
            max_image_size: Tamanho maximo para thumbnail opcional.

        Retorno:
            None: Apenas configura a instancia.
        """
        self.records = records
        self.max_image_size = max_image_size

    def __len__(self) -> int:
        """Retorna a quantidade de amostras.

        Parametros:
            Nenhum.

        Retorno:
            int: Total de registros carregados.
        """
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Carrega uma amostra individual do dataset.

        Parametros:
            index: Posicao numerica da amostra.

        Retorno:
            Dict[str, object]: Dicionario com imagem e metadados.
        """
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        if self.max_image_size:
            image.thumbnail((self.max_image_size, self.max_image_size))

        return {
            "fold": record.fold,
            "subset": record.subset,
            "path": record.path,
            "label_idx": record.label_idx,
            "raw_label_name": record.raw_label_name,
            "canonical_label": record.canonical_label,
            "image": image,
        }


def load_and_validate_manifest(manifest_path: Path, expected_fold: int) -> pd.DataFrame:
    """Carrega e valida o manifesto CSV de um fold.

    Parametros:
        manifest_path: Caminho do fold_manifest.csv.
        expected_fold: Fold esperado em todas as linhas.

    Retorno:
        pd.DataFrame: DataFrame validado.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto ausente: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing_columns = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in manifest.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {manifest_path}: {missing_columns}")

    if not (manifest["fold"] == expected_fold).all():
        raise ValueError(f"Manifesto {manifest_path} contem fold diferente de {expected_fold}")

    allowed_subsets = {"train", "val", "test"}
    unexpected_subsets = sorted(set(manifest["subset"].unique()) - allowed_subsets)
    if unexpected_subsets:
        raise ValueError(f"Subsets inesperados em {manifest_path}: {unexpected_subsets}")

    invalid_labels = sorted(set(manifest["label_name"].unique()) - set(RAW_TO_CANONICAL))
    if invalid_labels:
        raise ValueError(f"Classes inesperadas em {manifest_path}: {invalid_labels}")

    missing_paths = [path for path in manifest["path"].tolist() if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Foram encontrados paths inexistentes em {manifest_path}. Exemplo: {missing_paths[0]}"
        )

    return manifest


def assert_label_idx_alignment(manifest: pd.DataFrame) -> None:
    """Verifica o alinhamento entre label_idx e label_name.

    Parametros:
        manifest: Manifesto do fold.

    Retorno:
        None: Apenas valida e levanta excecao em caso de erro.
    """
    expected = {raw_label: idx for idx, raw_label in enumerate(RAW_LABEL_ORDER)}
    for row in manifest.itertuples(index=False):
        if int(row.label_idx) != expected[row.label_name]:
            raise ValueError(
                f"label_idx inconsistente para {row.path}: "
                f"esperado {expected[row.label_name]}, obtido {row.label_idx}"
            )


def _pilot_stratified_trim(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """Reduz um subset de forma aproximadamente estratificada.

    Parametros:
        frame: DataFrame de um unico subset.
        max_samples: Limite maximo de amostras.
        seed: Semente de reproducibilidade.

    Retorno:
        pd.DataFrame: Subconjunto reduzido.
    """
    if len(frame) <= max_samples:
        return frame

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    class_names = sorted(shuffled["label_name"].unique().tolist())
    base_quota = max_samples // len(class_names)
    remainder = max_samples % len(class_names)

    selected_parts = []
    for index, class_name in enumerate(class_names):
        class_frame = shuffled[shuffled["label_name"] == class_name]
        target = base_quota + (1 if index < remainder else 0)
        selected_parts.append(class_frame.head(target))

    trimmed = pd.concat(selected_parts, ignore_index=True)
    if len(trimmed) < max_samples:
        selected_paths = set(trimmed["path"].tolist())
        leftovers = shuffled[~shuffled["path"].isin(selected_paths)]
        trimmed = pd.concat([trimmed, leftovers.head(max_samples - len(trimmed))], ignore_index=True)

    return trimmed.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def subset_records(
    manifest: pd.DataFrame,
    subset_name: str,
    pilot_mode: Dict[str, object],
    seed: int,
) -> List[SampleRecord]:
    """Extrai registros de um subset com corte opcional de piloto.

    Parametros:
        manifest: Manifesto completo do fold.
        subset_name: Nome do subset desejado.
        pilot_mode: Bloco de configuracao do piloto.
        seed: Semente para reproducibilidade.

    Retorno:
        List[SampleRecord]: Lista tipada de registros.
    """
    frame = manifest[manifest["subset"] == subset_name].copy()
    limit_key = f"max_{subset_name}_samples_per_fold"
    if bool(pilot_mode.get("enabled")):
        frame = _pilot_stratified_trim(frame, int(pilot_mode[limit_key]), seed)

    return [
        SampleRecord(
            fold=int(row.fold),
            subset=row.subset,
            path=row.path,
            label_idx=int(row.label_idx),
            raw_label_name=row.label_name,
            canonical_label=canonicalize_dataset_label(row.label_name),
        )
        for row in frame.itertuples(index=False)
    ]


def dataset_summary(records: List[SampleRecord]) -> Dict[str, int]:
    """Conta amostras por rotulo canonico.

    Parametros:
        records: Lista de registros de um subset.

    Retorno:
        Dict[str, int]: Contagem por rotulo.
    """
    summary = {label: 0 for label in CANONICAL_LABELS}
    for record in records:
        summary[record.canonical_label] += 1
    return summary
