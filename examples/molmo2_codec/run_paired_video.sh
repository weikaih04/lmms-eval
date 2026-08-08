#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MANIFEST=${MANIFEST:-${ROOT}/examples/llava_onevision2_repro/manifests/VIDEO_QUALIFICATION_RANDOM200_V2_MANIFEST.json}

: "${SPLIT:?set SPLIT=short, medium, long, or mlvu}"
: "${MOLMO2_CODEC_REPO:?set MOLMO2_CODEC_REPO to the Molmo2 codec training repository}"
: "${MOLMO2_CODEC_PTOK:?set MOLMO2_CODEC_PTOK to the Stage-1 checkpoint}"
: "${MOLMO2_CODEC_STAGE2:?set MOLMO2_CODEC_STAGE2 to a consolidated Stage-2 checkpoint}"
: "${MOLMO2_CODEC_GAMMA:?set MOLMO2_CODEC_GAMMA to the fixed gamma artifact}"

PROTOCOL=${PROTOCOL:-molmo384}
NUM_PROCESSES=${NUM_PROCESSES:-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/out/molmo2_codec_paired_${PROTOCOL}}
CACHE_ROOT=${CACHE_ROOT:-${ROOT}/out/molmo2_codec_cache_8192}
SEQ_LEN=${SEQ_LEN:-65536}
VISUAL_TOKEN_BUDGET=${VISUAL_TOKEN_BUDGET:-8192}
BACKENDS=${BACKENDS:-dense,codec}

case "${SPLIT}" in
  short) TASK=ov2_videomme_short_wo_sutitle ;;
  medium) TASK=ov2_videomme_medium_wo_sutitle ;;
  long) TASK=ov2_videomme_long_wo_sutitle ;;
  mlvu) TASK=mlvu_dev ;;
  *) echo "unsupported SPLIT=${SPLIT}" >&2; exit 2 ;;
esac

case "${PROTOCOL}" in
  molmo384)
    FRAME_CAP=384
    TIMELINE_MODE=full_span_2fps
    ;;
  ov2_policy)
    TIMELINE_MODE=full_span_1fps
    case "${SPLIT}" in
      short) FRAME_CAP=128 ;;
      medium) FRAME_CAP=256 ;;
      long) FRAME_CAP=640 ;;
      mlvu) FRAME_CAP=512 ;;
    esac
    ;;
  *) echo "unsupported PROTOCOL=${PROTOCOL}" >&2; exit 2 ;;
esac

for path in "${MANIFEST}" "${MOLMO2_CODEC_PTOK}" "${MOLMO2_CODEC_STAGE2}" "${MOLMO2_CODEC_GAMMA}"; do
  test -s "${path}"
done
test -d "${MOLMO2_CODEC_REPO}"

export LMMS_EVAL_DOC_ID_MANIFEST=${MANIFEST}
export PYTHONPATH=${ROOT}:${MOLMO2_CODEC_REPO}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_ROOT}/${SPLIT}" \
  "${CACHE_ROOT}/gop_selected" "${CACHE_ROOT}/motion" "${CACHE_ROOT}/features"

IFS=',' read -r -a backend_list <<< "${BACKENDS}"
for backend in "${backend_list[@]}"; do
  case "${backend}" in dense|codec) ;; *) echo "unsupported backend=${backend}" >&2; exit 2 ;; esac
  output=${OUTPUT_ROOT}/${SPLIT}/${backend}
  trace=${OUTPUT_ROOT}/${SPLIT}/${backend}_trace.jsonl
  mkdir -p "${output}"
  model_args="training_repo=${MOLMO2_CODEC_REPO},pretrained=${MOLMO2_CODEC_STAGE2},ptok_checkpoint=${MOLMO2_CODEC_PTOK},gamma_artifact=${MOLMO2_CODEC_GAMMA},video_backend=${backend},p_variant=real,visual_token_budget=${VISUAL_TOKEN_BUDGET},timeline_max_frames=${FRAME_CAP},timeline_sampling_mode=${TIMELINE_MODE},max_frames=${FRAME_CAP},seq_len=${SEQ_LEN},gop_cache_dir=${CACHE_ROOT}/gop_selected,gop_cache_read_dirs=${CACHE_ROOT}/gop_selected,motion_cache_dir=${CACHE_ROOT}/motion,motion_cache_read_dirs=${CACHE_ROOT}/motion,feature_cache_dir=${CACHE_ROOT}/features,trace_output=${trace}"

  python -m accelerate.commands.launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    -m lmms_eval \
    --model molmo2_codec \
    --model_args "${model_args}" \
    --tasks "${TASK}" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output}"
  MAIN_PROCESS_PORT=$((MAIN_PROCESS_PORT + 1))
done
