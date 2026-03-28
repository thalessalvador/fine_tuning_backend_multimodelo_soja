from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_backend import FoldData, TrainFoldResult
from .transformers_backend import TransformersTrainingBackend


class UnslothTrainingBackend(TransformersTrainingBackend):
    """Backend que tenta usar Unsloth e cai para transformers quando preciso."""

    backend_name = "unsloth"

    def _unsloth_available(self) -> bool:
        """Verifica se o pacote Unsloth esta instalado.

        Parametros:
            Nenhum.

        Retorno:
            bool: True quando Unsloth for importavel.
        """
        return importlib.util.find_spec("unsloth") is not None

    def train_fold(
        self,
        fold_data: FoldData,
        output_dir: Path,
        config: Dict[str, Any],
        logger,
    ) -> TrainFoldResult:
        """Treina um fold priorizando Unsloth e usando fallback quando necessario.

        Parametros:
            fold_data: Estrutura com train, val e test.
            output_dir: Pasta do fold.
            config: Configuracao resolvida.
            logger: Logger ativo.

        Retorno:
            TrainFoldResult: Resultado do treino com backend efetivo registrado.
        """
        if not bool(config["backend"].get("enable_unsloth", False)):
            logger.warning("Backend unsloth desativado em config. Usando transformers.")
            result = super().train_fold(fold_data, output_dir, config, logger)
            result.runtime_metadata["backend_used"] = "transformers_fallback"
            result.runtime_metadata["fallback_reason"] = "unsloth_disabled"
            result.runtime_metadata["runtime_family"] = "transformers"
            return result

        if not self._unsloth_available():
            if not bool(config["backend"].get("enable_transformers_fallback", False)):
                raise RuntimeError("Unsloth nao esta instalado e o fallback em transformers esta desativado.")
            logger.warning("Unsloth nao encontrado no ambiente. Usando backend transformers como fallback.")
            result = super().train_fold(fold_data, output_dir, config, logger)
            result.runtime_metadata["backend_used"] = "transformers_fallback"
            result.runtime_metadata["fallback_reason"] = "unsloth_not_installed"
            result.runtime_metadata["runtime_family"] = "transformers"
            result.runtime_metadata["pilot_report"]["warnings"].append("unsloth_not_installed")
            return result

        logger.warning(
            "Implementacao v1 do backend unsloth ainda usa o fluxo compatibilizado via transformers. "
            "O ambiente foi marcado como apto para migracao incremental."
        )
        result = super().train_fold(fold_data, output_dir, config, logger)
        result.runtime_metadata["backend_used"] = "unsloth_compat_mode"
        result.runtime_metadata["fallback_reason"] = "v1_compat_mode"
        result.runtime_metadata["pilot_report"]["warnings"].append("unsloth_compat_mode")
        return result

    def predict(
        self,
        records: List[Any],
        adapter_dir: Optional[Path],
        output_dir: Path,
        config: Dict[str, Any],
        logger,
    ) -> pd.DataFrame:
        """Executa inferencia via fluxo compativel da v1.

        Parametros:
            records: Registros do subset.
            adapter_dir: Pasta do adapter salvo.
            output_dir: Pasta do fold.
            config: Configuracao resolvida.
            logger: Logger ativo.

        Retorno:
            pd.DataFrame: Predicoes padronizadas.
        """
        return super().predict(records, adapter_dir, output_dir, config, logger)
