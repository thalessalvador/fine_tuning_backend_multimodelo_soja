from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from .label_normalization import CANONICAL_LABELS, label_to_index, normalize_predicted_label


def build_user_message(prompt: str) -> List[Dict[str, object]]:
    """Monta a mensagem multimodal basica do usuario.

    Parametros:
        prompt: Instrucao textual de classificacao.

    Retorno:
        List[Dict[str, object]]: Estrutura no formato de chat multimodal.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def format_training_texts(processor, prompt: str, answer: str) -> Tuple[str, str]:
    """Gera textos de treino para o mascaramento correto do loss.

    Parametros:
        processor: Processor multimodal do modelo.
        prompt: Instrucao textual base.
        answer: Rotulo alvo da amostra.

    Retorno:
        Tuple[str, str]: Texto completo e texto somente do prompt.
    """
    user_messages = build_user_message(prompt)
    full_messages = user_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        }
    ]
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return full_text, prompt_text


def collate_for_training(batch: List[Dict[str, object]], processor, prompt: str) -> Dict[str, torch.Tensor]:
    """Converte um batch em tensores com labels mascarados.

    Parametros:
        batch: Lista de amostras vindas do dataset.
        processor: Processor multimodal do modelo.
        prompt: Prompt textual padronizado.

    Retorno:
        Dict[str, torch.Tensor]: Tensores prontos para o treino supervisionado.
    """
    images = [item["image"] for item in batch]
    labels = [item["canonical_label"] for item in batch]
    full_texts: List[str] = []
    prompt_texts: List[str] = []

    for answer in labels:
        full_text, prompt_text = format_training_texts(processor, prompt, answer)
        full_texts.append(full_text)
        prompt_texts.append(prompt_text)

    model_inputs = processor(
        text=full_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=prompt_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    masked_labels = model_inputs["input_ids"].clone()
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
    for row_index, prompt_len in enumerate(prompt_lengths.tolist()):
        masked_labels[row_index, :prompt_len] = -100

    masked_labels[model_inputs["attention_mask"] == 0] = -100
    model_inputs["labels"] = masked_labels
    return model_inputs


def parse_generated_label(decoded_text: str) -> Optional[str]:
    """Extrai um rotulo canonico de uma resposta textual ou JSON.

    Parametros:
        decoded_text: Texto bruto decodificado da saida do modelo.

    Retorno:
        Optional[str]: Rotulo canonico ou None.
    """
    try:
        parsed = json.loads(decoded_text)
        if isinstance(parsed, dict) and "predicted_label" in parsed:
            return normalize_predicted_label(str(parsed["predicted_label"]))
    except json.JSONDecodeError:
        pass

    return normalize_predicted_label(decoded_text)


def build_prediction_rows(
    batch: List[Dict[str, object]],
    raw_outputs: List[str],
    normalized_labels: List[Optional[str]],
) -> List[Dict[str, object]]:
    """Converte uma saida de inferencia em linhas serializaveis.

    Parametros:
        batch: Lista de amostras de entrada.
        raw_outputs: Respostas textuais brutas geradas pelo modelo.
        normalized_labels: Rotulos canonicos normalizados.

    Retorno:
        List[Dict[str, object]]: Linhas prontas para virar DataFrame.
    """
    rows: List[Dict[str, object]] = []
    for index, item in enumerate(batch):
        true_label = str(item["canonical_label"])
        normalized = normalized_labels[index] or "__INVALID__"
        rows.append(
            {
                "fold": int(item["fold"]),
                "path": item["path"],
                "true_idx": label_to_index(true_label),
                "true_label": true_label,
                "pred_idx": label_to_index(normalized) if normalized in CANONICAL_LABELS else -1,
                "pred_label": normalized if normalized in CANONICAL_LABELS else "",
                "correct": normalized == true_label,
                "raw_response": raw_outputs[index],
                "error": "" if normalized in CANONICAL_LABELS else "invalid_label",
            }
        )
    return rows


def predictions_from_rows(rows: List[Dict[str, object]]) -> pd.DataFrame:
    """Converte linhas de predicao em DataFrame padronizado.

    Parametros:
        rows: Linhas de predicao serializaveis.

    Retorno:
        pd.DataFrame: Tabela de predicoes padronizada.
    """
    return pd.DataFrame(rows)
