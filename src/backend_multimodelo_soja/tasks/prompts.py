from __future__ import annotations

from typing import Dict


DEFAULT_TRAIN_PROMPT = (
    "Classifique este grao de soja em exatamente uma das classes: "
    "Broken, Immature, Intact, Skin Damaged, Spotted. "
    "Responda apenas com o rotulo."
)


def resolve_prompt(training_config: Dict[str, object]) -> str:
    """Resolve o prompt padrao ou customizado do experimento.

    Parametros:
        training_config: Bloco de configuracao de treino.

    Retorno:
        str: Prompt final a ser usado em treino e avaliacao.
    """
    prompt = str(training_config.get("train_prompt", "")).strip()
    return prompt or DEFAULT_TRAIN_PROMPT
