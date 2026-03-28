from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FoldData:
    """Agrupa os subsets de um fold para treino e avaliacao.

    Parametros:
        fold: Indice do fold.
        train_records: Registros de treino.
        val_records: Registros de validacao.
        test_records: Registros de teste.
    """

    fold: int
    train_records: List[Any]
    val_records: List[Any]
    test_records: List[Any]


@dataclass
class TrainFoldResult:
    """Resultado do treino de um fold em um backend.

    Parametros:
        best_epoch: Melhor epoca selecionada.
        best_val_macro_f1: Melhor macro F1 em validacao.
        history: Historico serializavel do treino.
        val_metrics_best: Metricas do melhor ponto de validacao.
        adapter_dir: Pasta onde o adapter foi salvo.
        runtime_metadata: Metadados de runtime do backend.
    """

    best_epoch: int
    best_val_macro_f1: float
    history: Dict[str, Any]
    val_metrics_best: Dict[str, Any]
    adapter_dir: Path
    runtime_metadata: Dict[str, Any]


class BaseTrainingBackend(ABC):
    """Contrato base para qualquer backend de treino multimodelo."""

    @abstractmethod
    def train_fold(
        self,
        fold_data: FoldData,
        output_dir: Path,
        config: Dict[str, Any],
        logger,
    ) -> TrainFoldResult:
        """Treina um fold e salva o melhor adapter.

        Parametros:
            fold_data: Estrutura com subsets do fold.
            output_dir: Pasta do fold.
            config: Configuracao resolvida da run.
            logger: Logger ativo.

        Retorno:
            TrainFoldResult: Resultado do treino com melhor ponto selecionado.
        """

    @abstractmethod
    def predict(
        self,
        records: List[Any],
        adapter_dir: Optional[Path],
        output_dir: Path,
        config: Dict[str, Any],
        logger,
    ) -> pd.DataFrame:
        """Executa inferencia para um conjunto de registros.

        Parametros:
            records: Registros do subset a avaliar.
            adapter_dir: Pasta do adapter treinado.
            output_dir: Pasta do fold.
            config: Configuracao resolvida da run.
            logger: Logger ativo.

        Retorno:
            pd.DataFrame: Predicoes padronizadas.
        """

    @abstractmethod
    def save_backend_metadata(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        runtime_metadata: Dict[str, Any],
    ) -> Path:
        """Salva metadados de runtime do backend.

        Parametros:
            output_dir: Pasta do fold.
            config: Configuracao resolvida.
            runtime_metadata: Metadados do backend.

        Retorno:
            Path: Caminho do arquivo salvo.
        """
