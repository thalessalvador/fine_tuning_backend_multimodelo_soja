from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from src.backend_multimodelo_soja.experiment.logging_utils import setup_logger
from src.backend_multimodelo_soja.backends.transformers_backend import select_torch_dtype


def parse_args():
    """Processa os argumentos da preparacao de artefatos para Ollama."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Diretorio da run de fine-tuning.")
    parser.add_argument(
        "--folds",
        nargs="*",
        type=int,
        help="Lista de folds a exportar. Se omitido, exporta todos os fold_* encontrados.",
    )
    parser.add_argument(
        "--ollama-base-model",
        required=True,
        help="Nome do modelo base no Ollama usado no FROM do Modelfile.",
    )
    parser.add_argument(
        "--ollama-model-template",
        default="soja-{run_name}-fold{fold}",
        help="Template do nome do modelo no Ollama. Suporta {run_name} e {fold}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.15,
        help="Temperatura a registrar no Modelfile.",
    )
    parser.add_argument(
        "--export-strategy",
        default="adapter",
        choices=["adapter", "merge_gguf"],
        help="Estrategia de exportacao. Hoje adapter esta implementada; merge_gguf fica preparado como trilha futura.",
    )
    parser.add_argument(
        "--merge-device",
        default="cpu",
        choices=["cpu", "auto", "cuda"],
        help="Dispositivo preferido para carregar e mergear o modelo base com o adapter.",
    )
    parser.add_argument(
        "--llama-cpp-convert-script",
        help="Caminho para convert_hf_to_gguf.py do llama.cpp.",
    )
    parser.add_argument(
        "--llama-cpp-quantize-bin",
        help="Caminho para o executavel de quantizacao do llama.cpp.",
    )
    parser.add_argument(
        "--gguf-outtype",
        default="f16",
        help="Outtype usado em convert_hf_to_gguf.py, por exemplo f16 ou bf16.",
    )
    parser.add_argument(
        "--gguf-quant-type",
        default="Q4_K_M",
        help="Tipo de quantizacao do llama.cpp, por exemplo Q4_K_M.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Quando informado, executa ollama create para cada fold exportado.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    """Le um arquivo JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_fold_dirs(run_dir: Path, requested_folds: Iterable[int] | None) -> List[Path]:
    """Resolve quais folds devem ser processados."""
    if requested_folds:
        fold_dirs = [run_dir / f"fold_{fold}" for fold in requested_folds]
    else:
        fold_dirs = sorted(
            candidate
            for candidate in run_dir.iterdir()
            if candidate.is_dir() and candidate.name.startswith("fold_")
        )

    missing = [fold_dir for fold_dir in fold_dirs if not fold_dir.exists()]
    if missing:
        missing_str = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(f"Folds inexistentes para exportacao: {missing_str}")
    return fold_dirs


def fold_index_from_dir(fold_dir: Path) -> int:
    """Extrai o indice do fold a partir do nome da pasta."""
    return int(fold_dir.name.split("_", maxsplit=1)[1])


def build_modelfile_text(
    from_reference: str,
    parameter_lines: List[str] | None = None,
) -> str:
    """Monta o texto do Modelfile."""
    lines = [f"FROM {from_reference}"]
    if parameter_lines:
        lines.extend(parameter_lines)
    lines.append("")
    return "\n".join(lines)


def resolve_merge_device(merge_device: str) -> str | None:
    """Resolve o device_map para a fase de merge."""
    if merge_device == "cpu":
        return None
    if merge_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("merge_device=cuda foi solicitado, mas CUDA nao esta disponivel.")
        return "cuda:0"
    return "auto" if torch.cuda.is_available() else None


def merge_adapter_with_base_model(
    base_model_id: str,
    adapter_dir: Path,
    merged_output_dir: Path,
    bf16_if_available: bool,
    merge_device: str,
    logger,
) -> Dict[str, Any]:
    """Faz merge do adapter LoRA com o modelo base e salva o modelo HF resultante."""
    torch_dtype = select_torch_dtype(bf16_if_available)
    device_map = resolve_merge_device(merge_device)

    logger.info(
        "Iniciando merge HF do adapter com o modelo base: model=%s device=%s dtype=%s",
        base_model_id,
        merge_device,
        str(torch_dtype).replace("torch.", ""),
    )

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch_dtype,
    }
    if device_map == "auto":
        model_kwargs["device_map"] = "auto"
    elif device_map == "cuda:0":
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForImageTextToText.from_pretrained(base_model_id, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged_model = model.merge_and_unload()

    merged_output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    processor.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    tokenizer_artifacts = sorted(
        path.name
        for path in merged_output_dir.iterdir()
        if path.is_file() and path.name.startswith("tokenizer")
    )

    return {
        "merged_output_dir": str(merged_output_dir.resolve()),
        "dtype": str(torch_dtype).replace("torch.", ""),
        "merge_device": merge_device,
        "tokenizer_artifacts": tokenizer_artifacts,
    }


def run_subprocess(command: List[str], logger, step_name: str) -> Dict[str, Any]:
    """Executa um subprocesso e retorna metadados serializaveis."""
    logger.info("Executando %s: %s", step_name, command)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    payload = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode != 0:
        logger.warning("%s falhou com exit code %s", step_name, result.returncode)
    return payload


def export_fold(
    fold_dir: Path,
    run_dir: Path,
    ollama_base_model: str,
    ollama_model_template: str,
    temperature: float,
    export_strategy: str,
    merge_device: str,
    llama_cpp_convert_script: str | None,
    llama_cpp_quantize_bin: str | None,
    gguf_outtype: str,
    gguf_quant_type: str,
    create_model: bool,
    logger,
) -> Dict[str, Any]:
    """Prepara os artefatos de um fold para consumo pelo Ollama."""
    adapter_dir = fold_dir / "adapter"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    adapter_config_path = adapter_dir / "adapter_config.json"
    backend_metadata_path = fold_dir / "backend_metadata.json"
    train_config_path = fold_dir / "train_config_resolved.json"
    val_metrics_path = fold_dir / "val_metrics_best.json"

    required_paths = [
        adapter_dir,
        adapter_weights_path,
        adapter_config_path,
        backend_metadata_path,
        train_config_path,
        val_metrics_path,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_str = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(f"Fold {fold_dir.name} sem artefatos suficientes para exportacao: {missing_str}")

    train_config = read_json(train_config_path)
    backend_metadata = read_json(backend_metadata_path)
    val_metrics = read_json(val_metrics_path)

    fold = fold_index_from_dir(fold_dir)
    run_name = run_dir.name
    ollama_model_name = ollama_model_template.format(run_name=run_name, fold=fold)

    export_manifest = {
        "fold": fold,
        "run_dir": str(run_dir.resolve()),
        "fold_dir": str(fold_dir.resolve()),
        "base_model_id": train_config["base_model_id"],
        "ollama_base_model": ollama_base_model,
        "ollama_model_name": ollama_model_name,
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_weights_path": str(adapter_weights_path.resolve()),
        "export_strategy": export_strategy,
        "status": "prepared",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": train_config.get("resolved", {}).get("best_epoch"),
        "best_val_macro_f1": train_config.get("resolved", {}).get("best_val_macro_f1"),
        "val_metrics_best": val_metrics,
        "backend_metadata": {
            "backend_requested": backend_metadata.get("backend_requested"),
            "backend_used": backend_metadata.get("backend_used"),
            "runtime_family": backend_metadata.get("runtime_family"),
            "model_family": backend_metadata.get("model_family"),
            "quantization_mode": backend_metadata.get("quantization_mode"),
        },
        "notes": [],
    }

    export_dir = fold_dir / "ollama_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    if export_strategy == "adapter":
        export_adapter_weights_path = export_dir / "adapter_model.safetensors"
        export_adapter_config_path = export_dir / "adapter_config.json"
        shutil.copy2(adapter_weights_path, export_adapter_weights_path)
        shutil.copy2(adapter_config_path, export_adapter_config_path)
        modelfile_path = export_dir / "Modelfile"
        modelfile_path.write_text(
            build_modelfile_text(
                from_reference=ollama_base_model,
                parameter_lines=[
                    "ADAPTER ./adapter_model.safetensors",
                    f"PARAMETER temperature {temperature}",
                ],
            ),
            encoding="utf-8",
        )
        export_manifest["export_adapter_weights_path"] = str(export_adapter_weights_path.resolve())
        export_manifest["export_adapter_config_path"] = str(export_adapter_config_path.resolve())
        export_manifest["modelfile_path"] = str(modelfile_path.resolve())
        export_manifest["notes"] = [
            "Artefato preparado para consumo pelo projeto de inferencia via Ollama.",
            "Este manifesto nao garante compatibilidade da arquitetura com ADAPTER; ele apenas materializa o caminho de importacao.",
        ]

        if create_model:
            logger.info("Executando ollama create para %s", ollama_model_name)
            result = subprocess.run(
                ["ollama", "create", ollama_model_name, "-f", str(modelfile_path.resolve())],
                check=False,
                capture_output=True,
                text=True,
            )
            export_manifest["ollama_create"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            export_manifest["status"] = "created" if result.returncode == 0 else "create_failed"
            if result.returncode != 0:
                logger.warning("ollama create falhou para %s com exit code %s", ollama_model_name, result.returncode)
    else:
        merged_hf_dir = export_dir / "merged_hf_model"
        merge_report = merge_adapter_with_base_model(
            base_model_id=train_config["base_model_id"],
            adapter_dir=adapter_dir,
            merged_output_dir=merged_hf_dir,
            bf16_if_available=bool(train_config.get("memory", {}).get("bf16_if_available", True)),
            merge_device=merge_device,
            logger=logger,
        )
        export_manifest["merge_report"] = merge_report
        export_manifest["merged_hf_dir"] = str(merged_hf_dir.resolve())
        export_manifest["notes"] = [
            "A estrategia merge_gguf faz merge do adapter com o modelo base antes do deploy no Ollama.",
            "Quando ferramentas do llama.cpp sao fornecidas, o script tenta converter e quantizar para GGUF.",
        ]

        gguf_source_reference = "./merged_hf_model"
        export_manifest["status"] = "merged_hf_ready"

        if llama_cpp_convert_script:
            merged_gguf_path = export_dir / "merged_model.gguf"
            convert_result = run_subprocess(
                [
                    "python",
                    str(Path(llama_cpp_convert_script).resolve()),
                    str(merged_hf_dir.resolve()),
                    "--outfile",
                    str(merged_gguf_path.resolve()),
                    "--outtype",
                    gguf_outtype,
                ],
                logger,
                "convert_hf_to_gguf",
            )
            export_manifest["gguf_conversion"] = convert_result
            if convert_result["returncode"] == 0 and merged_gguf_path.exists():
                export_manifest["merged_gguf_path"] = str(merged_gguf_path.resolve())
                gguf_source_reference = "./merged_model.gguf"
                export_manifest["status"] = "gguf_ready"

                if llama_cpp_quantize_bin:
                    quantized_gguf_path = export_dir / f"merged_model.{gguf_quant_type}.gguf"
                    quantize_result = run_subprocess(
                        [
                            str(Path(llama_cpp_quantize_bin).resolve()),
                            str(merged_gguf_path.resolve()),
                            str(quantized_gguf_path.resolve()),
                            gguf_quant_type,
                        ],
                        logger,
                        "quantize_gguf",
                    )
                    export_manifest["gguf_quantization"] = quantize_result
                    if quantize_result["returncode"] == 0 and quantized_gguf_path.exists():
                        export_manifest["quantized_gguf_path"] = str(quantized_gguf_path.resolve())
                        gguf_source_reference = f"./{quantized_gguf_path.name}"
                        export_manifest["status"] = "quantized_gguf_ready"

        modelfile_path = export_dir / "Modelfile"
        modelfile_path.write_text(
            build_modelfile_text(
                from_reference=gguf_source_reference,
                parameter_lines=[f"PARAMETER temperature {temperature}"],
            ),
            encoding="utf-8",
        )
        export_manifest["modelfile_path"] = str(modelfile_path.resolve())

        if create_model:
            logger.info("Executando ollama create para %s", ollama_model_name)
            result = subprocess.run(
                ["ollama", "create", ollama_model_name, "-f", str(modelfile_path.resolve())],
                check=False,
                capture_output=True,
                text=True,
            )
            export_manifest["ollama_create"] = {
                "command": ["ollama", "create", ollama_model_name, "-f", str(modelfile_path.resolve())],
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            export_manifest["status"] = "created" if result.returncode == 0 else "create_failed"
            if result.returncode != 0:
                logger.warning("ollama create falhou para %s com exit code %s", ollama_model_name, result.returncode)

    export_manifest_path = export_dir / "ollama_export.json"
    with export_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(export_manifest, handle, ensure_ascii=False, indent=2)

    logger.info(
        "Fold %s preparado para Ollama: model=%s status=%s",
        fold,
        ollama_model_name,
        export_manifest["status"],
    )
    return export_manifest


def main():
    """Orquestra a preparacao de artefatos de exportacao para Ollama."""
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Diretorio de run inexistente: {run_dir}")

    logger = setup_logger(run_dir / "ollama_export.log", "prepare_ollama_artifacts")
    logger.info("Preparando artefatos de exportacao para Ollama em %s", run_dir)

    fold_dirs = discover_fold_dirs(run_dir, args.folds)
    manifests = [
        export_fold(
            fold_dir=fold_dir,
            run_dir=run_dir,
            ollama_base_model=args.ollama_base_model,
            ollama_model_template=args.ollama_model_template,
            temperature=float(args.temperature),
            export_strategy=str(args.export_strategy),
            merge_device=str(args.merge_device),
            llama_cpp_convert_script=args.llama_cpp_convert_script,
            llama_cpp_quantize_bin=args.llama_cpp_quantize_bin,
            gguf_outtype=str(args.gguf_outtype),
            gguf_quant_type=str(args.gguf_quant_type),
            create_model=bool(args.create),
            logger=logger,
        )
        for fold_dir in fold_dirs
    ]

    summary = {
        "run_dir": str(run_dir),
        "ollama_base_model": args.ollama_base_model,
        "model_name_template": args.ollama_model_template,
        "export_strategy": args.export_strategy,
        "merge_device": args.merge_device,
        "llama_cpp_convert_script": args.llama_cpp_convert_script,
        "llama_cpp_quantize_bin": args.llama_cpp_quantize_bin,
        "gguf_outtype": args.gguf_outtype,
        "gguf_quant_type": args.gguf_quant_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folds": manifests,
    }
    summary_path = run_dir / "ollama_exports_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    logger.info("Resumo de exportacao salvo em %s", summary_path)


if __name__ == "__main__":
    main()
