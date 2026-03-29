from __future__ import annotations

import copy
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from ..experiment.data import SoyManifestDataset
from ..experiment.metrics import compute_classification_metrics
from ..tasks.label_normalization import CANONICAL_LABELS, RAW_LABEL_ORDER
from ..tasks.prompts import resolve_prompt
from ..tasks.sft_classification import (
    build_prediction_rows,
    build_user_message,
    collate_for_training,
    parse_generated_label,
    predictions_from_rows,
)
from .base_backend import BaseTrainingBackend, FoldData, TrainFoldResult


def _safe_package_version(package_name: str) -> Optional[str]:
    """Consulta a versao de um pacote quando instalado.

    Parametros:
        package_name: Nome do pacote Python.

    Retorno:
        Optional[str]: Versao instalada ou None.
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def select_torch_dtype(bf16_if_available: bool) -> torch.dtype:
    """Seleciona o dtype padrao para o hardware atual.

    Parametros:
        bf16_if_available: Indica se bf16 deve ser priorizado.

    Retorno:
        torch.dtype: Dtype selecionado.
    """
    if bf16_if_available and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def model_float_dtype(model) -> torch.dtype:
    """Obtem o dtype float dominante do modelo.

    Parametros:
        model: Modelo ja carregado.

    Retorno:
        torch.dtype: Dtype float dominante.
    """
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def move_inputs_to_device(
    model_inputs: Dict[str, torch.Tensor],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Move tensores para o dispositivo e ajusta dtypes float.

    Parametros:
        model_inputs: Dicionario de tensores.
        device: Dispositivo alvo.
        model_dtype: Dtype float principal do modelo.

    Retorno:
        Dict[str, torch.Tensor]: Entradas prontas para uso.
    """
    moved = {}
    for key, value in model_inputs.items():
        tensor = value.to(device)
        if torch.is_floating_point(tensor):
            tensor = tensor.to(model_dtype)
        moved[key] = tensor
    return moved


def current_vram_gb() -> float:
    """Consulta o pico atual de VRAM alocada.

    Parametros:
        Nenhum.

    Retorno:
        float: VRAM maxima alocada em gigabytes.
    """
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 3))


DEFAULT_LORA_TARGET_MODULES = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]

FAMILY_EXTRA_LORA_TARGET_MODULES = {
    'ministral3': ['linear_1', 'linear_2', 'merging_layer'],
}


def _candidate_lora_target_modules(config: Dict[str, Any]) -> List[str]:
    """Resolve a lista desejada de modulos alvo para LoRA.

    Parametros:
        config: Configuracao resolvida da run.

    Retorno:
        List[str]: Sufixos de modulos desejados, vindos do override ou do padrao dinamico.
    """
    override = config['lora'].get('target_modules')
    if override:
        return [str(item).strip() for item in override if str(item).strip()]

    model_family = str(config['backend'].get('model_family', '')).strip().lower()
    resolved = list(DEFAULT_LORA_TARGET_MODULES)
    resolved.extend(FAMILY_EXTRA_LORA_TARGET_MODULES.get(model_family, []))
    return resolved


def _module_bucket(module_name: str) -> str:
    """Classifica um modulo em texto, visao, projector ou outro.

    Parametros:
        module_name: Nome completo do modulo.

    Retorno:
        str: Categoria resumida do modulo.
    """
    lower = module_name.lower()
    if 'vision_tower' in lower or '.vision_' in lower or lower.startswith('vision_'):
        return 'vision'
    if 'multi_modal_projector' in lower or 'mm_projector' in lower or 'projector' in lower:
        return 'projector'
    if 'language_model' in lower or 'text_model' in lower or 'lm_head' in lower:
        return 'text'
    return 'other'


def resolve_lora_target_modules(model, config: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """Descobre dinamicamente quais sufixos de modulo LoRA existem no modelo.

    Parametros:
        model: Modelo ja carregado.
        config: Configuracao resolvida da run.

    Retorno:
        Tuple[List[str], Dict[str, Any]]: Sufixos resolvidos e relatorio resumido de cobertura.
    """
    desired_suffixes = _candidate_lora_target_modules(config)
    linear_module_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    matched_suffixes: List[str] = []
    matched_names: List[str] = []
    coverage = {'text': 0, 'vision': 0, 'projector': 0, 'other': 0}

    for suffix in desired_suffixes:
        suffix_matches = [
            name for name in linear_module_names if name == suffix or name.endswith(f'.{suffix}')
        ]
        if suffix_matches:
            matched_suffixes.append(suffix)
            matched_names.extend(suffix_matches)
            for module_name in suffix_matches:
                coverage[_module_bucket(module_name)] += 1

    report = {
        'requested_suffixes': desired_suffixes,
        'resolved_suffixes': matched_suffixes,
        'matched_module_count': len(matched_names),
        'coverage': coverage,
        'sample_modules': matched_names[:20],
    }
    return matched_suffixes, report


class TransformersTrainingBackend(BaseTrainingBackend):
    """Backend multimodal baseado em transformers, PEFT e bitsandbytes."""

    backend_name = "transformers"

    def _build_runtime_metadata(self, config: Dict[str, Any], used_backend: Optional[str] = None) -> Dict[str, Any]:
        """Monta metadados de runtime do backend.

        Parametros:
            config: Configuracao resolvida da run.
            used_backend: Nome efetivo do backend utilizado.

        Retorno:
            Dict[str, Any]: Metadados serializaveis de runtime.
        """
        dtype = str(select_torch_dtype(bool(config["memory"].get("bf16_if_available", True)))).replace("torch.", "")
        return {
            "backend_requested": config["backend"]["name"],
            "backend_used": used_backend or self.backend_name,
            "runtime_family": config["backend"]["runtime_family"],
            "model_family": config["backend"]["model_family"],
            "model_id": config["base_model_id"],
            "quantization_mode": config["backend"]["quantization_mode"],
            "device_map": "auto" if torch.cuda.is_available() else None,
            "main_dtype": dtype,
            "torch": _safe_package_version("torch"),
            "transformers": _safe_package_version("transformers"),
            "peft": _safe_package_version("peft"),
            "bitsandbytes": _safe_package_version("bitsandbytes"),
            "unsloth": _safe_package_version("unsloth"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gradient_checkpointing": bool(config["memory"].get("gradient_checkpointing", False)),
        }

    def _load_processor_and_model(self, config: Dict[str, Any], apply_lora: bool = True):
        """Carrega processor e modelo base com LoRA ou QLoRA.

        Parametros:
            config: Configuracao resolvida.
            apply_lora: Indica se a injecao de LoRA deve ser aplicada.

        Retorno:
            tuple: Processor e modelo prontos para treino.
        """
        memory = config["memory"]
        torch_dtype = select_torch_dtype(bool(memory.get("bf16_if_available", True)))

        quantization_config = None
        if bool(memory.get("load_in_4bit", False)) or str(config["backend"]["quantization_mode"]).lower() == "qlora_4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        processor = AutoProcessor.from_pretrained(config["base_model_id"], trust_remote_code=True)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": torch_dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForImageTextToText.from_pretrained(config["base_model_id"], **model_kwargs)

        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        if apply_lora and bool(config['lora'].get('enabled', True)):
            target_modules, resolution_report = resolve_lora_target_modules(model, config)
            if not target_modules:
                raise ValueError(
                    'Nenhum modulo LoRA compativel foi encontrado no modelo para os sufixos desejados.'
                )
            lora_config = LoraConfig(
                r=int(config['lora']['r']),
                lora_alpha=int(config['lora']['alpha']),
                lora_dropout=float(config['lora']['dropout']),
                target_modules=target_modules,
                bias='none',
                task_type='CAUSAL_LM',
            )
            model = get_peft_model(model, lora_config)
            setattr(model, '_lora_resolution_report', resolution_report)

        if bool(memory.get('gradient_checkpointing', False)):
            model.gradient_checkpointing_enable()
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            if hasattr(model, 'config'):
                model.config.use_cache = False

        resolution_report = getattr(model, '_lora_resolution_report', None)
        if resolution_report is not None and hasattr(model, 'config'):
            setattr(model.config, 'lora_resolution_report', resolution_report)

        return processor, model

    def _run_epoch(
        self,
        model,
        dataloader: DataLoader,
        optimizer: AdamW,
        grad_accum_steps: int,
        device: torch.device,
        logger,
        epoch: int,
        logging_steps: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Executa uma epoca completa de treino.

        Parametros:
            model: Modelo multimodal em treino.
            dataloader: DataLoader do subset de treino.
            optimizer: Otimizador do treino.
            grad_accum_steps: Passos de acumulacao de gradiente.
            device: Dispositivo de execucao.
            logger: Logger ativo.
            epoch: Numero da epoca.
            logging_steps: Intervalo de log.

        Retorno:
            Tuple[float, Dict[str, float]]: Loss media e estatisticas de runtime.
        """
        model.train()
        model_dtype = model_float_dtype(model)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        resolution_report = getattr(getattr(model, 'config', None), 'lora_resolution_report', None)
        if resolution_report:
            logger.info(
                'lora_resolution requested=%s resolved=%s coverage=%s matched_module_count=%s',
                resolution_report['requested_suffixes'],
                resolution_report['resolved_suffixes'],
                resolution_report['coverage'],
                resolution_report['matched_module_count'],
            )

        step_times: List[float] = []
        start_epoch = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step, batch in enumerate(dataloader, start=1):
            step_start = time.perf_counter()
            batch = move_inputs_to_device(batch, device, model_dtype)
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            total_loss += loss.item()

            if step % grad_accum_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step_duration = time.perf_counter() - step_start
            step_times.append(step_duration)
            if step % logging_steps == 0 or step == len(dataloader):
                logger.info(
                    "fold epoch=%s step=%s/%s loss=%.6f step_time_s=%.3f max_vram_gb=%.3f",
                    epoch,
                    step,
                    len(dataloader),
                    total_loss / step,
                    step_duration,
                    current_vram_gb(),
                )

        epoch_duration = time.perf_counter() - start_epoch
        return total_loss / max(len(dataloader), 1), {
            "step_time_mean_s": sum(step_times) / max(len(step_times), 1),
            "epoch_time_s": epoch_duration,
            "max_vram_gb": current_vram_gb(),
            "cpu_offload_detected": False,
        }

    @torch.inference_mode()
    def _predict_with_model(
        self,
        model,
        processor,
        dataset: SoyManifestDataset,
        batch_size: int,
        prompt: str,
        max_new_tokens: int,
        device: torch.device,
    ) -> pd.DataFrame:
        """Executa inferencia em um dataset com o modelo informado.

        Parametros:
            model: Modelo carregado.
            processor: Processor multimodal.
            dataset: Dataset do subset.
            batch_size: Tamanho do batch de avaliacao.
            prompt: Prompt textual padronizado.
            max_new_tokens: Limite de tokens novos.
            device: Dispositivo de execucao.

        Retorno:
            pd.DataFrame: Predicoes padronizadas.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda items: items)
        rows = []
        was_training = model.training
        model.eval()
        try:
            for batch in dataloader:
                images = [item["image"] for item in batch]
                prompt_messages = build_user_message(prompt)
                prompt_text = processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts = [prompt_text for _ in batch]
                model_inputs = processor(
                    text=prompts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                model_inputs = move_inputs_to_device(model_inputs, device, model_float_dtype(model))
                generation_kwargs = {
                    **model_inputs,
                    'max_new_tokens': max_new_tokens,
                    'do_sample': False,
                }
                if getattr(model, 'generation_config', None) is not None:
                    generation_config = copy.deepcopy(model.generation_config)
                    generation_config.max_length = None
                    generation_kwargs['generation_config'] = generation_config
                generated_ids = model.generate(**generation_kwargs)
                input_lengths = model_inputs["attention_mask"].sum(dim=1)
                raw_outputs = []
                normalized_labels = []
                for row_index in range(generated_ids.size(0)):
                    generated_tail = generated_ids[row_index, input_lengths[row_index] :]
                    decoded = processor.decode(generated_tail, skip_special_tokens=True).strip()
                    raw_outputs.append(decoded)
                    normalized_labels.append(parse_generated_label(decoded))
                rows.extend(build_prediction_rows(batch, raw_outputs, normalized_labels))
        finally:
            if was_training:
                model.train()
        return predictions_from_rows(rows)

    def train_fold(
        self,
        fold_data: FoldData,
        output_dir: Path,
        config: Dict[str, Any],
        logger,
    ) -> TrainFoldResult:
        """Treina um fold inteiro usando o backend transformers.

        Parametros:
            fold_data: Estrutura com train, val e test do fold.
            output_dir: Pasta do fold.
            config: Configuracao resolvida.
            logger: Logger ativo do fold.

        Retorno:
            TrainFoldResult: Melhor ponto salvo e metadados do treino.
        """
        max_image_size = int(config["memory"]["max_image_size"])
        train_dataset = SoyManifestDataset(fold_data.train_records, max_image_size=max_image_size)
        val_dataset = SoyManifestDataset(fold_data.val_records, max_image_size=max_image_size)

        processor, model = self._load_processor_and_model(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            model.to(device)

        prompt = resolve_prompt(config["training"])
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["training"]["per_device_train_batch_size"]),
            shuffle=True,
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=lambda batch: collate_for_training(batch, processor=processor, prompt=prompt),
        )
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        history: Dict[str, Any] = {
            "fold": fold_data.fold,
            "train_samples": len(fold_data.train_records),
            "val_samples": len(fold_data.val_records),
            "test_samples": len(fold_data.test_records),
            "epochs": [],
        }
        best_val_macro_f1 = -1.0
        best_epoch = -1
        best_metrics: Dict[str, Any] = {}
        adapter_dir = output_dir / "adapter"
        pilot_runtime: Dict[str, Any] = {}

        for epoch in range(1, int(config["training"]["num_epochs"]) + 1):
            train_loss, train_runtime = self._run_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                grad_accum_steps=int(config["training"]["gradient_accumulation_steps"]),
                device=device,
                logger=logger,
                epoch=epoch,
                logging_steps=int(config["training"]["logging_steps"]),
            )
            val_predictions = self._predict_with_model(
                model=model,
                processor=processor,
                dataset=val_dataset,
                batch_size=int(config["training"]["per_device_eval_batch_size"]),
                prompt=prompt,
                max_new_tokens=int(config["training"]["max_new_tokens"]),
                device=device,
            )
            val_metrics = self._metrics_from_predictions(val_predictions, fold_data.fold, str(config["base_model_id"]))
            val_metrics["split"] = "val"
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = train_loss
            val_metrics["runtime"] = train_runtime
            history["epochs"].append(val_metrics)
            pilot_runtime = train_runtime

            logger.info(
                "fold=%s epoch=%s train_loss=%.6f val_macro_f1=%.6f val_accuracy=%.6f val_failures=%s epoch_time_s=%.2f",
                fold_data.fold,
                epoch,
                train_loss,
                val_metrics["macro_f1"],
                val_metrics["accuracy"],
                val_metrics["inference_failures"],
                train_runtime["epoch_time_s"],
            )

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                best_metrics = val_metrics
                model.save_pretrained(adapter_dir)
                processor.save_pretrained(adapter_dir)

        if best_epoch < 0:
            raise RuntimeError(f"Nenhum checkpoint valido foi selecionado no fold {fold_data.fold}")

        runtime_metadata = self._build_runtime_metadata(config)
        runtime_metadata["pilot_report"] = {
            "enabled": bool(config["pilot_mode"]["enabled"]),
            "fold": fold_data.fold,
            "backend_used": runtime_metadata["backend_used"],
            "max_vram_gb": pilot_runtime.get("max_vram_gb", 0.0),
            "step_time_mean_s": pilot_runtime.get("step_time_mean_s", 0.0),
            "epoch_time_s": pilot_runtime.get("epoch_time_s", 0.0),
            "cpu_offload_detected": pilot_runtime.get("cpu_offload_detected", False),
            "warnings": [],
        }

        return TrainFoldResult(
            best_epoch=best_epoch,
            best_val_macro_f1=best_val_macro_f1,
            history=history,
            val_metrics_best=best_metrics,
            adapter_dir=adapter_dir,
            runtime_metadata=runtime_metadata,
        )

    def _metrics_from_predictions(self, predictions: pd.DataFrame, fold: int, model_name: str) -> Dict[str, object]:
        """Deriva metricas a partir do DataFrame de predicoes.

        Parametros:
            predictions: Tabela de predicoes.
            fold: Fold avaliado.
            model_name: Nome do modelo.

        Retorno:
            Dict[str, object]: Estrutura serializavel de metricas.
        """
        y_true = predictions["true_label"].tolist()
        y_pred = [
            pred if pred in CANONICAL_LABELS else CANONICAL_LABELS[0]
            for pred in predictions["pred_label"].replace("", pd.NA).fillna("__INVALID__").tolist()
        ]
        inference_failures = int((predictions["error"] != "").sum())
        return compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels_for_report=CANONICAL_LABELS,
            raw_labels_for_per_class=RAW_LABEL_ORDER,
            fold=fold,
            model_name=model_name,
            inference_failures=inference_failures,
        )

    def _reload_best_adapter(self, config: Dict[str, Any], adapter_dir: Path):
        """Recarrega o modelo base e aplica o adapter salvo.

        Parametros:
            config: Configuracao resolvida.
            adapter_dir: Pasta do adapter salvo.

        Retorno:
            tuple: Processor, modelo e dispositivo prontos para inferencia.
        """
        processor, model = self._load_processor_and_model(config, apply_lora=False)
        if adapter_dir.exists():
            model = PeftModel.from_pretrained(model, adapter_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            model.to(device)
        model.eval()
        return processor, model, device

    def predict(
        self,
        records: List[Any],
        adapter_dir: Optional[Path],
        output_dir: Path,
        config: Dict[str, Any],
        logger,
    ) -> pd.DataFrame:
        """Executa predicoes com o melhor adapter salvo.

        Parametros:
            records: Registros do subset a avaliar.
            adapter_dir: Pasta do adapter salvo.
            output_dir: Pasta do fold.
            config: Configuracao resolvida.
            logger: Logger ativo.

        Retorno:
            pd.DataFrame: Predicoes padronizadas.
        """
        del output_dir, logger
        dataset = SoyManifestDataset(records, max_image_size=int(config["memory"]["max_image_size"]))
        processor, model, device = self._reload_best_adapter(config, adapter_dir or Path())
        prompt = resolve_prompt(config["training"])
        return self._predict_with_model(
            model=model,
            processor=processor,
            dataset=dataset,
            batch_size=int(config["training"]["per_device_eval_batch_size"]),
            prompt=prompt,
            max_new_tokens=int(config["training"]["max_new_tokens"]),
            device=device,
        )

    def save_backend_metadata(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        runtime_metadata: Dict[str, Any],
    ) -> Path:
        """Salva metadados do backend em JSON.

        Parametros:
            output_dir: Pasta do fold.
            config: Configuracao resolvida.
            runtime_metadata: Metadados do runtime.

        Retorno:
            Path: Caminho do arquivo salvo.
        """
        metadata = {
            **runtime_metadata,
            "backend_section": config["backend"],
        }
        output_path = output_dir / "backend_metadata.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return output_path
