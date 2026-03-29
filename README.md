# Backend Modular Multimodelo para Fine-Tuning de Soja

Projeto de fine-tuning supervisionado multimodal com backend modular para classificacao de graos de soja. Esta implementacao separa a orquestracao experimental da camada de runtime de treino e inferencia para permitir trocar a familia de backend com menor atrito tecnico, preservando a comparabilidade metodologica com os manifests e artefatos do projeto anterior.

## Status atual do projeto

O caminho estavel e suportado deste projeto hoje e o backend `transformers`. As tentativas de estabilizar um backend nativo em `unsloth` foram interrompidas nesta versao por incompatibilidades praticas observadas no ambiente de execucao usado no desenvolvimento, uma `NVIDIA GeForce RTX 5070 Laptop GPU` com stack recente de CUDA/PyTorch.

O suporte a `unsloth` fica tratado como trabalho futuro. A intencao era reduzir VRAM e tempo por step para abrir espaco a testes com modelos maiores, inclusive variantes de 8B. Na pratica, isso nao se confirmou com robustez suficiente para manter o recurso como caminho principal do projeto.

### Tentativas realizadas com Unsloth

- execucao nativa no Windows com PyTorch nightly e CUDA 12.8
- execucao em Docker com a imagem oficial da Unsloth
- ajuste de import order para priorizar o patching do Unsloth
- testes com `device_map` automatico e forcado em GPU
- testes com diferentes combinacoes de gradient checkpointing
- fallback para `transformers` apos falha do backend nativo

### Problemas encontrados

- falha recorrente no backward do caminho nativo da Unsloth: `RuntimeError: Function MmBackward0 returned an invalid gradient at index 1 - expected device meta but got cuda:0`
- inconsistencias de device no fallback apos tentativa nativa: mistura de tensores em CPU e `cuda:0`
- warnings e incompatibilidades do stack Triton/Unsloth em Blackwell, incluindo `Failed to import Triton kernels`
- ausencia de ganho pratico de VRAM e throughput em relacao ao baseline estavel em `transformers`
- em alguns testes, aumento de consumo e degradacao de performance, com passos significativamente mais lentos do que o baseline anterior

### Decisao desta versao

- manter `transformers` como backend padrao e suportado
- preservar a comparabilidade experimental com os folds externos e com o pipeline anterior
- documentar `unsloth` apenas como trilha futura, e nao como caminho operacional desta versao


## Objetivo do projeto

- reutilizar exatamente os `fold_manifest.csv` ja produzidos pelo experimento anterior
- manter treino por fold, validacao em `val` e avaliacao final em `test`
- permitir alternancia entre backend principal e backend fallback por configuracao
- gerar artefatos equivalentes aos do pipeline anterior
- registrar metadados de runtime para tornar os experimentos auditaveis

## Motivacao para backend modular

O projeto anterior atende bem como pipeline experimental controlado, mas o acoplamento com detalhes de tokenizer, processador, quantizacao e runtime dificulta a troca de familia de VLM. Nesta nova raiz, a camada experimental continua simples e comparavel, enquanto o backend fica encapsulado em modulos registraveis.

## Diferenca entre este projeto e o pipeline anterior

- este projeto vive integralmente em `C:\Projetos\Pos_ia\TCC\fine_tuning_backend_multimodelo_soja`
- o projeto antigo em `C:\Projetos\Pos_ia\TCC\fine_tuning_qwen3_vl_8B_soja` e apenas referencia de leitura
- a comparabilidade metodologica e preservada pelo reuso dos manifests externos, pelos mesmos subsets `train`, `val` e `test`, pelo mesmo conjunto de metricas e pelos mesmos artefatos agregados
- a camada de treino usa hoje `transformers` como backend estavel e deixa a trilha `unsloth` como trabalho futuro

## Referencias externas

- Repositorio tecnico de referencia: `https://github.com/Mindful-AI-Assistants/Fine-tuning-Ministral-3-with-Unsloth`
- Documentacao de referencia da Unsloth para Ministral 3: `https://unsloth.ai/docs/new/ministral-3`
- Repositorio metodologico de referencia para `source_experiment_dir`: `https://github.com/thalessalvador/ensaio_classif_soja_llm_multimodais`

Esta implementacao nao copia o repositorio externo. Ela apenas organiza a arquitetura local para permitir adotar a stack da Unsloth de forma incremental, sem perder o contrato experimental da tese.

## Relacao com `source_experiment_dir` e com o projeto de inferencia base

A configuracao `source_experiment_dir` faz referencia direta ao projeto `ensaio_classif_soja_llm_multimodais`. O uso correto deste campo e metodologicamente importante quando se deseja comparar o modelo base nao fine-tunado com o modelo fine-tunado, inclusive em cenarios posteriores de inferencia com o modelo tunado servido em Ollama.

Fluxo recomendado:

1. Rode primeiro, no repositorio `https://github.com/thalessalvador/ensaio_classif_soja_llm_multimodais`, o experimento do modelo base desejado.
2. Gere la os folds completos e os respectivos `fold_manifest.csv`.
3. Aponte aqui o `source_experiment_dir` para a pasta de saida daquele experimento ja concluido.
4. Reaproveite exatamente os mesmos `fold_manifest.csv` neste projeto de fine-tuning.
5. Treine e avalie o modelo fine-tunado usando os mesmos subsets `train`, `val` e `test` herdados do experimento base.

Esse procedimento e o que garante comparacao justa e evita `data leakage`, porque o split usado pelo modelo fine-tunado continua sendo exatamente o mesmo split ja congelado no experimento do modelo nao fine-tunado. Se depois voce quiser comparar a inferencia do modelo base com a inferencia do modelo tunado dentro do Ollama, a referencia de folds continua alinhada e auditavel.

## Arquitetura em duas camadas

### Camada 1: orquestracao experimental

Arquivos principais:

- `train_model_fold.py`
- `run_model_cv.py`
- `aggregate_model_cv.py`
- `src/backend_multimodelo_soja/experiment/`

Responsabilidades:

- carregar e validar configuracao
- localizar manifests existentes por fold
- separar `train`, `val` e `test`
- executar folds em sequencia
- salvar logs, metricas e tabelas agregadas

### Camada 2: backend de treino e inferencia

Arquivos principais:

- `src/backend_multimodelo_soja/backends/base_backend.py`
- `src/backend_multimodelo_soja/backends/transformers_backend.py`
- `src/backend_multimodelo_soja/backends/backend_registry.py`

Responsabilidades:

- carregar processor e modelo conforme a familia
- aplicar LoRA ou QLoRA com resolucao dinamica dos modulos alvo
- executar treino supervisionado
- selecionar melhor epoca por `val_macro_f1`
- recarregar adapter e avaliar `test`
- registrar `backend_metadata.json`

## Estrutura do projeto

```text
finetune_config.yaml
train_model_fold.py
run_model_cv.py
aggregate_model_cv.py
requirements.txt
README.md
input_manifests/
outputs/
references/
src/
  backend_multimodelo_soja/
    experiment/
    backends/
    tasks/
```

## Requisitos de ambiente

Este projeto depende de PyTorch com suporte compativel com a sua GPU e o seu driver. O `requirements.txt` nao instala `torch` automaticamente para evitar conflito entre builds CUDA.

### 1. Verificar GPU e driver

```powershell
nvidia-smi
```

### 2. Verificar Python

```powershell
python --version
```

### 3. Criar e ativar o ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativacao:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Atualizar o pip

```powershell
python -m pip install --upgrade pip
```

### 5. Instalar PyTorch

CUDA 12.4:

```powershell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

CUDA 12.1:

```powershell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

CPU apenas:

```powershell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### 6. Instalar as dependencias do projeto

```powershell
pip install -r requirements.txt
```

## Observacoes de compatibilidade de CUDA e GPUs novas

Se a GPU for muito nova e o build estavel do PyTorch nao reconhecer a arquitetura, troque para nightly:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Se o modelo base for da familia Ministral 3, e prudente atualizar tambem:

```powershell
pip install --upgrade mistral-common
pip install --upgrade --pre transformers
```

## Como configurar o backend por modelo

O arquivo [finetune_config.yaml](/c:/Projetos/Pos_ia/TCC/fine_tuning_backend_multimodelo_soja/finetune_config.yaml) concentra a configuracao do runtime.

### Uso atual

O caminho suportado nesta versao do projeto e:

```yaml
base_model_id: mistralai/Ministral-3-3B-Instruct-2512-BF16
backend:
  name: transformers
  runtime_family: transformers
  model_family: ministral3
  quantization_mode: qlora_4bit
  enable_unsloth: false
  enable_transformers_fallback: true
  extra_requirements_profile: transformers-cu128
```

### Papel de cada campo

- `base_model_id`: checkpoint exato a ser carregado.
- `backend.name`: backend solicitado pela orquestracao.
- `backend.runtime_family`: familia logica do runtime efetivo.
- `backend.model_family`: familia logica do modelo para regras especificas.
- `backend.quantization_mode`: estrategia principal de quantizacao.
- `backend.enable_unsloth`: mantido apenas por compatibilidade historica; o caminho estavel atual nao depende dele.
- `backend.enable_transformers_fallback`: permite cair para `transformers`.
- `backend.extra_requirements_profile`: perfil sugerido de dependencias da maquina.

Regra pratica: `model_family` nao substitui `base_model_id`.
`model_family` diz como tratar a familia; `base_model_id` diz qual checkpoint concreto carregar.

### Como o LoRA escolhe os modulos

Por padrao, este projeto nao depende mais de uma lista fixa de `target_modules` no YAML. O backend inspeciona o modelo carregado e resolve dinamicamente quais sufixos de modulo existem de fato para aquela arquitetura.

Comportamento atual:

- usa um conjunto base de sufixos comuns a transformers causais, como `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` e `down_proj`
- adiciona extras conhecidos por familia quando fizer sentido, como o projector multimodal do `Ministral 3`
- filtra automaticamente apenas os sufixos que realmente existem no modelo carregado
- registra no log um resumo da resolucao, incluindo cobertura em texto, visao e projector

O campo `lora.target_modules` continua disponivel apenas como override manual. Ele deve ser usado somente quando voce quiser fixar explicitamente o escopo do experimento.

### Familias mapeadas no projeto

As familias abaixo continuam documentadas porque o codigo e o YAML contemplam esses `model_family`, mas isso nao significa que todas estejam homologadas no ambiente atual.

| Familia | Exemplo de `base_model_id` | `model_family` | Status nesta versao |
| --- | --- | --- | --- |
| Ministral 3 3B | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | `ministral3` | Caminho estavel atual |
| Ministral 3 8B | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `ministral3` | Referencia futura; validar conforme VRAM |
| Llama 3.x | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `llama3` | Referencia futura |
| Llama 4 | `unsloth/Llama-4-Scout-17B-16E-Instruct` | `llama4` | Referencia futura |
| Qwen 2.5 texto | `Qwen/Qwen2.5-7B-Instruct` | `qwen25` | Referencia futura |
| Qwen 3 texto | `Qwen/Qwen3-8B` | `qwen3` | Referencia futura |
| Qwen 2.5 VL | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen25vl` | Referencia futura |
| Qwen 3 VL | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3vl` | Referencia futura |
| Gemma 3 4B | `google/gemma-3-4b-it` | `gemma3` | Referencia futura |
| Gemma 3 12B | `google/gemma-3-12b-it` | `gemma3` | Referencia futura |
| DeepSeek VL | `deepseek-ai/deepseek-vl2-small` | `deepseek_vl` | Mapeado para `transformers` |
| DeepSeek OCR | `deepseek-ai/DeepSeek-OCR-2` | `deepseek_ocr` | Mapeado para `transformers` |
| Phi 3 Vision | `microsoft/Phi-3.5-vision-instruct` | `phi3_vision` | Mapeado para `transformers` |

### Regras de decisao rapida

- Para esta versao, use `transformers` como backend operacional.
- Em modelos de visao, confirme sempre se o checkpoint escolhido e realmente `vision` ou `VL`, e nao a variante texto apenas.
- Se trocar de familia, troque `base_model_id` e `backend.model_family` juntos.
- Trate modelos fora do baseline atual como referencia futura ate que sejam validados em run propria.

## Como o projeto preserva comparabilidade metodologica

- le os manifests prontos do experimento anterior via `source_experiment_dir`
- nao recalcula folds
- mantem a mesma ordem canonica de labels
- usa `train` apenas para treino, `val` para selecao do melhor adapter e `test` para avaliacao final
- gera os mesmos artefatos centrais por fold e na agregacao completa

## Formato esperado dos manifests

Cada `fold_manifest.csv` externo precisa conter:

- `fold`
- `subset`
- `path`
- `label_idx`
- `label_name`

O pipeline falha explicitamente se:

- faltar coluna obrigatoria
- existir `path` inexistente
- houver classe fora do conjunto esperado
- `label_idx` nao estiver alinhado com a ordem padrao

## Como rodar o piloto de um fold

O piloto recomendado usa apenas `fold_0`, subset reduzido e `1` epoca. Ajuste `pilot_mode.enabled: true` no YAML antes de rodar, porque o arquivo atual pode estar configurado para execucao completa.

### 1. Executar o fold piloto

```powershell
python train_model_fold.py --config finetune_config.yaml --fold 0
```

O diretorio da run e criado automaticamente em `outputs/`, com nome derivado de modelo, backend, timestamp e modo.

### 2. Conferir os artefatos do fold

Arquivos gerados dentro de `outputs/<modelo>_<backend>_AAAAMMDD_HHMMSS_pilot_fold0/fold_0/`:

- `run_config.json`
- `runtime.log`
- `train_config_resolved.json`
- `train_history.json`
- `val_metrics_best.json`
- `predictions.csv`
- `metrics.json`
- `confusion_matrix.csv`
- `adapter/`
- `backend_metadata.json`

O arquivo `backend_metadata.json` registra:

- backend solicitado e backend efetivamente usado
- versoes de `torch`, `transformers`, `peft`, `bitsandbytes` e `unsloth`
- `quantization_mode`
- `model_family`
- `device_map`
- `main_dtype`
- relatorio do piloto com `max_vram_gb`, `step_time_mean_s`, `epoch_time_s` e warnings

## Como rodar a CV completa

Quando o piloto estiver estavel, desative o corte reduzido:

```yaml
pilot_mode:
  enabled: false
```

Depois rode:

```powershell
python run_model_cv.py --config finetune_config.yaml
```

O runner:

- cria uma nova pasta em `outputs/<modelo>_<backend>_AAAAMMDD_HHMMSS_cv_full`
- executa os folds `0..4` em sequencia
- aborta se qualquer fold falhar
- roda a agregacao final ao final da CV

### Como retomar uma run existente a partir de um fold especifico

Se um fold intermediario falhar ou a maquina reiniciar, voce pode retomar a mesma pasta mae sem perder os folds ja concluidos.

Fluxo recomendado:

1. Apague manualmente a pasta do fold incompleto, por exemplo `fold_1`.
2. Reexecute a CV informando a pasta mae da run e o fold de retomada.

Exemplo:

```powershell
python run_model_cv.py --config finetune_config.yaml --run-dir outputs/ministral_3_3b_instruct_2512_bf16_transformers_20260328_132513_cv_full --start-fold 1
```

Esse comando:

- preserva os folds ja concluidos
- executa apenas do fold indicado em diante
- roda a agregacao final ao terminar

Regra importante: use `--start-fold > 0` apenas com `--run-dir`, para retomar uma run ja existente.

## Onde localizar adapters e artefatos

Por fold:

- `adapter/`: adapter salvo do melhor ponto de validacao
- `predictions.csv`: predicoes normalizadas do subset `test`
- `metrics.json`: metricas finais do fold
- `val_metrics_best.json`: metrica da melhor epoca em `val`
- `backend_metadata.json`: metadados do runtime

Por run completa:

- `run_config.json`
- `runtime.log`
- `aggregate.log`
- `cv_summary.json`
- `cv_summary.csv`
- `cv_summary_per_fold.csv`
- `thesis_summary_table.csv`
- `thesis_folds_table.csv`

## Como manter comparabilidade com `ensaio_classif_soja_llm_multimodais`

1. Nao altere os manifests externos.
2. Nao escreva nada em `source_experiment_dir`.
3. Mantenha `label_set` exatamente igual ao conjunto canonico.
4. Compare as tabelas `thesis_summary_table.csv` e `thesis_folds_table.csv` com a run de referencia.
5. Preserve o uso de `test` apenas na avaliacao final.

## Problemas comuns e comandos de diagnostico

### Verificar se o PyTorch ve a GPU

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Verificar se o pacote Unsloth esta instalado

```powershell
python -c "import importlib.util; print(importlib.util.find_spec('unsloth') is not None)"
```

### Verificar se o config resolve caminhos corretamente

```powershell
python -c "from pathlib import Path; from src.backend_multimodelo_soja.experiment.config import load_config; import json; cfg = load_config(Path('finetune_config.yaml')); print(json.dumps({'dataset_root': cfg['dataset_root'], 'source_experiment_dir': cfg['source_experiment_dir'], 'output_root': cfg['output_root']}, indent=2))"
```

### Rodar apenas a agregacao de uma run ja concluida

```powershell
python aggregate_model_cv.py --run-dir outputs/ministral_3_3b_instruct_2512_bf16_transformers_20260328_132513_cv_full
```

## Estado atual da v1

- `transformers_backend.py` e o caminho funcional e suportado de treino e inferencia
- a orquestracao permanece pronta para evolucao futura por backend, mas esta versao deve ser lida como baseline `transformers`
- qualquer retomada da trilha `unsloth` deve ocorrer como trabalho futuro, em branch ou versao separada, sem contaminar o baseline estavel

## Observacao operacional final

Este projeto nao substitui automaticamente o projeto anterior. Ele funciona como backend modular de treino comparavel, enquanto a inferencia posterior com prompts JSON pode continuar no pipeline antigo usando o adapter treinado aqui.
