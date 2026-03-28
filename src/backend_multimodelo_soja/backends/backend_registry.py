from __future__ import annotations

from typing import Dict, Type

from .base_backend import BaseTrainingBackend
from .transformers_backend import TransformersTrainingBackend
from .unsloth_backend import UnslothTrainingBackend


BACKEND_REGISTRY: Dict[str, Type[BaseTrainingBackend]] = {
    "transformers": TransformersTrainingBackend,
    "unsloth": UnslothTrainingBackend,
}


def create_backend(name: str) -> BaseTrainingBackend:
    """Instancia um backend a partir do nome configurado.

    Parametros:
        name: Nome do backend registrado.

    Retorno:
        BaseTrainingBackend: Instancia do backend escolhido.
    """
    normalized = name.strip().lower()
    if normalized not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(f"Backend desconhecido: {name}. Disponiveis: {available}")
    return BACKEND_REGISTRY[normalized]()
