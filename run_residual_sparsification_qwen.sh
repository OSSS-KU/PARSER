#!/usr/bin/env bash
export NUMEXPR_MAX_THREADS=32
export PYTHONPATH=$PYTHONPATH:$(pwd)/.cache/huggingface/modules

set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

step_num=0
step() {
  step_num=$((step_num + 1))
  log ""
  log "############################################################"
  log "#"
  log "#  STEP ${step_num}: $*"
  log "#"
  log "############################################################"
}

run_cmd() {
  log "Running: $*"
  "$@"
}

trap 'log "ERROR at line ${LINENO}: ${BASH_COMMAND}"' ERR
log "residual sparsification pipeline start"

MODEL="Qwen/Qwen1.5-MoE-A2.7B"
DEVICE="cuda"
DEVICE_MAP="auto"
MODEL_DTYPE="bf16"

BLOCKS_DIR="auto"
COMPRESSED_DIR_ROOT="auto"
EVAL_DIR_ROOT="auto"

CALIBRATION_DATASET="dolly"
CALIBRATION_SAMPLES="512"
CALIBRATION_SEQ_LEN="2048"
CALIBRATION_BATCH="64"
CALIBRATION_PADDING="longest"
CALIBRATION_MAX_EXPERT_SAMPLES="2048"
CALIBRATION_SEED="0"

CALIBRATION_STREAMING="0"
CALIBRATION_SHUFFLE_BUFFER="50000"
CALIBRATION_PACKED="1"
CALIBRATION_EXCLUDE_DOMAINS=""

METHODS="original,parser-l-sp"
SPARSITIES="0.9,0.8,0.7"
LAYERS="last:16"
BACKBONE="barycenter"
MIN_UNITS="4"
PARSER_IMPORTANCE="formula"
PARSER_ROUTING_SCORE="0"

EVAL_TASKS="arc_easy,arc_challenge,winogrande,hellaswag,openbookqa,piqa,mmlu"
EVAL_BATCH="64"
EVAL_FEWSHOT="0"
EVAL_LIMIT=""
CLEANUP_COMPRESSED="1"

log "Config: MODEL=${MODEL} DEVICE=${DEVICE:-<default>} DEVICE_MAP=${DEVICE_MAP:-<none>} DTYPE=${MODEL_DTYPE:-<default>} METHODS=${METHODS} SPARSITIES=${SPARSITIES} EVAL_TASKS=${EVAL_TASKS}"
log "Config: BLOCKS_DIR=${BLOCKS_DIR} COMPRESSED_DIR_ROOT=${COMPRESSED_DIR_ROOT} EVAL_DIR_ROOT=${EVAL_DIR_ROOT}"
log "Config: LAYERS=${LAYERS}"
log "Config: CALIBRATION_DATASET=${CALIBRATION_DATASET} CALIBRATION_SAMPLES=${CALIBRATION_SAMPLES} CALIBRATION_SEQ_LEN=${CALIBRATION_SEQ_LEN} CALIBRATION_BATCH=${CALIBRATION_BATCH} CALIBRATION_PADDING=${CALIBRATION_PADDING} CALIBRATION_MAX_EXPERT_SAMPLES=${CALIBRATION_MAX_EXPERT_SAMPLES} CALIBRATION_SEED=${CALIBRATION_SEED} CALIBRATION_STREAMING=${CALIBRATION_STREAMING} CALIBRATION_SHUFFLE_BUFFER=${CALIBRATION_SHUFFLE_BUFFER:-<auto>} CALIBRATION_PACKED=${CALIBRATION_PACKED} CALIBRATION_EXCLUDE_DOMAINS=${CALIBRATION_EXCLUDE_DOMAINS:-<none>}"
log "Config: BACKBONE=${BACKBONE} MIN_UNITS=${MIN_UNITS} PARSER_IMPORTANCE=${PARSER_IMPORTANCE} PARSER_ROUTING_SCORE=${PARSER_ROUTING_SCORE}"
log "Config: EVAL_BATCH=${EVAL_BATCH} EVAL_FEWSHOT=${EVAL_FEWSHOT} EVAL_LIMIT=${EVAL_LIMIT:-<none>}"

if [[ "${BLOCKS_DIR}" == "auto" ]]; then
  log "Resolving BLOCKS_DIR from model: ${MODEL}"
  BLOCKS_DIR="$(python - "${MODEL}" <<'PY'
import sys
from moe_pipeline.output_paths import default_blocks_dir
print(default_blocks_dir(sys.argv[1]))
PY
)"
  log "Resolved BLOCKS_DIR=${BLOCKS_DIR}"
fi

IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
need_parser=0
for method_raw in "${METHOD_LIST[@]}"; do
  method="$(echo "${method_raw}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
  if [[ "${method}" == parser* ]]; then
    need_parser=1
  fi
done
log "Methods parsed: ${METHOD_LIST[*]}"
log "Calibration flags: need_parser=${need_parser}"

device_args=()
if [[ -n "${DEVICE}" ]]; then
  device_args=(--device "${DEVICE}")
fi
dtype_args=()
if [[ -n "${MODEL_DTYPE}" ]]; then
  dtype_args=(--dtype "${MODEL_DTYPE}")
fi
eval_args=()
if [[ -n "${EVAL_BATCH}" ]]; then
  eval_args+=(--batch-size "${EVAL_BATCH}")
fi
if [[ -n "${EVAL_FEWSHOT}" ]]; then
  eval_args+=(--num-fewshot "${EVAL_FEWSHOT}")
fi
if [[ -n "${EVAL_LIMIT}" ]]; then
  eval_args+=(--limit "${EVAL_LIMIT}")
fi
device_map_args=()
if [[ -n "${DEVICE_MAP}" ]]; then
  device_map_args=(--device-map "${DEVICE_MAP}")
fi

save_args=(python save_blocks.py \
  --model "${MODEL}" \
  --output-dir "${BLOCKS_DIR}" \
  "${device_args[@]}" \
  "${device_map_args[@]}" \
  "${dtype_args[@]}" \
  --skip-existing \
  --calibration-dataset "${CALIBRATION_DATASET}" \
  --calibration-samples "${CALIBRATION_SAMPLES}" \
  --calibration-seq-len "${CALIBRATION_SEQ_LEN}" \
  --calibration-batch-size "${CALIBRATION_BATCH}" \
  --calibration-padding "${CALIBRATION_PADDING}" \
  --calibration-max-expert-samples "${CALIBRATION_MAX_EXPERT_SAMPLES}" \
  --calibration-seed "${CALIBRATION_SEED}")
save_args+=(--timing-methods "${METHODS}")

if [[ "${CALIBRATION_PACKED}" == "1" ]]; then
  save_args+=(--calibration-packed)
fi

if [[ -n "${CALIBRATION_EXCLUDE_DOMAINS}" ]]; then
  save_args+=(--calibration-exclude-domains "${CALIBRATION_EXCLUDE_DOMAINS}")
fi

if [[ "${CALIBRATION_STREAMING}" == "1" ]]; then
  save_args+=(--calibration-streaming)
fi
if [[ -n "${CALIBRATION_SHUFFLE_BUFFER}" ]]; then
  save_args+=(--calibration-shuffle-buffer "${CALIBRATION_SHUFFLE_BUFFER}")
fi

if [[ "${need_parser}" == "1" ]]; then
  save_args+=(--calibrate)
fi
step "Save blocks (calibration + block extraction)"
run_cmd "${save_args[@]}"

IFS=',' read -r -a SPARSITY_LIST <<< "${SPARSITIES}"
IFS=',' read -r -a EVAL_TASK_LIST <<< "${EVAL_TASKS}"

EVAL_TASK_ARGS=()
for task_raw in "${EVAL_TASK_LIST[@]}"; do
  task="$(echo "${task_raw}" | tr -d '[:space:]')"
  if [[ -n "${task}" ]]; then
    EVAL_TASK_ARGS+=("${task}")
  fi
done
if [[ "${#EVAL_TASK_ARGS[@]}" -eq 0 ]]; then
  EVAL_TASK_ARGS=("wikitext")
fi
log "Evaluation tasks resolved: ${EVAL_TASK_ARGS[*]}"

for method_raw in "${METHOD_LIST[@]}"; do
  method="$(echo "${method_raw}" | tr -d '[:space:]')"
  method_norm="$(echo "${method}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "${method_norm}" ]]; then
    continue
  fi
  sparsity_idx=0
  for sparsity_raw in "${SPARSITY_LIST[@]}"; do
    sparsity="$(echo "${sparsity_raw}" | tr -d '[:space:]')"
    if [[ -z "${sparsity}" ]]; then
      continue
    fi

    if [[ "${method_norm}" == "original" && "${sparsity_idx}" -ne 0 ]]; then
      log "Skipping method=original at sparsity=${sparsity} (already evaluated for first sparsity in list)"
      sparsity_idx=$((sparsity_idx + 1))
      continue
    fi
    sparsity_idx=$((sparsity_idx + 1))

    tag_method="${method_norm}"
    tag_residual_mode="na"
    tag_residual_scope="na"
    method_importance="formula"
    method_parser_routing_score="0"
    if [[ "${method_norm}" == parser-* ]]; then
      method_importance="${PARSER_IMPORTANCE}"
      method_parser_routing_score="${PARSER_ROUTING_SCORE}"
    fi

    log "Resolving compression tag for method=${tag_method} sparsity=${sparsity} importance=${method_importance} parser_routing_score=${method_parser_routing_score}"
    tag="$(python - "${tag_method}" "${sparsity}" "${tag_residual_mode}" "${tag_residual_scope}" "${BACKBONE}" "${method_importance}" "${method_parser_routing_score}" <<'PY'
import sys
from moe_pipeline.output_paths import compression_tag
parser_routing = str(sys.argv[7]).strip().lower() in {"1", "true", "yes", "on"}
print(
    compression_tag(
        method=sys.argv[1],
        sparsity=float(sys.argv[2]),
        residual_mode=sys.argv[3],
        residual_scope=sys.argv[4],
        backbone=sys.argv[5],
        importance=sys.argv[6],
        parser_routing_score=parser_routing if sys.argv[1].startswith("parser-") else None,
    )
)
PY
)"
    log "Compression tag: ${tag}"

    if [[ "${COMPRESSED_DIR_ROOT}" == "auto" ]]; then
      log "Resolving COMPRESSED_DIR (auto)"
      COMPRESSED_DIR="$(python - "${BLOCKS_DIR}" "${tag_method}" "${sparsity}" "${tag_residual_mode}" "${tag_residual_scope}" "${BACKBONE}" "${method_importance}" "${method_parser_routing_score}" <<'PY'
import sys
from pathlib import Path
from moe_pipeline.output_paths import default_compressed_dir
parser_routing = str(sys.argv[8]).strip().lower() in {"1", "true", "yes", "on"}
print(
    default_compressed_dir(
        Path(sys.argv[1]),
        method=sys.argv[2],
        sparsity=float(sys.argv[3]),
        residual_mode=sys.argv[4],
        residual_scope=sys.argv[5],
        backbone=sys.argv[6],
        importance=sys.argv[7],
        parser_routing_score=parser_routing if sys.argv[2].startswith("parser-") else None,
    )
)
PY
)"
    else
      COMPRESSED_DIR="${COMPRESSED_DIR_ROOT}/${tag}/compressed"
    fi
    log "COMPRESSED_DIR=${COMPRESSED_DIR}"

    if [[ "${EVAL_DIR_ROOT}" == "auto" ]]; then
      log "Resolving EVAL_DIR (auto)"
      EVAL_DIR="$(python - "${BLOCKS_DIR}" "${COMPRESSED_DIR}" <<'PY'
import sys
from pathlib import Path
from moe_pipeline.output_paths import default_eval_dir
print(
    default_eval_dir(
        compressed_dir=Path(sys.argv[2]),
        blocks_dir=Path(sys.argv[1]),
    )
)
PY
)"
    else
      EVAL_DIR="${EVAL_DIR_ROOT}/${tag}/eval"
    fi
    log "EVAL_DIR=${EVAL_DIR}"

    step "Compress blocks (method=${method_norm} sparsity=${sparsity})"
    run_cmd python compress_blocks.py \
      --blocks-dir "${BLOCKS_DIR}" \
      --output-dir "${COMPRESSED_DIR}" \
      "${device_args[@]}" \
      --layers "${LAYERS}" \
      --method "${method_norm}" \
      --sparsity "${sparsity}" \
      --backbone "${BACKBONE}" \
      --importance "${method_importance}" \
      --parser-routing-score "${method_parser_routing_score}" \
      --min-units "${MIN_UNITS}" \
      --skip-existing

    step "Evaluate blocks (method=${method_norm} sparsity=${sparsity})"
    run_cmd python evaluate_blocks.py \
      --model "${MODEL}" \
      --blocks-dir "${BLOCKS_DIR}" \
      --compressed-dir "${COMPRESSED_DIR}" \
      --output-dir "${EVAL_DIR}" \
      "${device_args[@]}" \
      "${device_map_args[@]}" \
      "${dtype_args[@]}" \
      "${eval_args[@]}" \
      --layers "${LAYERS}" \
      --tasks "${EVAL_TASK_ARGS[@]}"

    if [[ "${CLEANUP_COMPRESSED}" == "1" ]]; then
      step "Cleanup compressed checkpoints (method=${method_norm} sparsity=${sparsity})"
      if [[ -d "${COMPRESSED_DIR}" ]]; then
        run_cmd find "${COMPRESSED_DIR}" -maxdepth 1 -type f -name 'layer_*.pt' -delete
      else
        log "Compressed dir not found; skipping cleanup."
      fi
    fi
  done
done
log "residual sparsification pipeline done"
