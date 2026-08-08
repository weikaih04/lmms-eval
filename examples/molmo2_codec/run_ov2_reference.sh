#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MANIFEST=${MANIFEST:-${ROOT}/examples/llava_onevision2_repro/manifests/VIDEO_QUALIFICATION_RANDOM200_V2_MANIFEST.json}

: "${SPLIT:?set SPLIT=short, medium, long, or mlvu}"
PROTOCOL=${PROTOCOL:-official}
MODEL=${MODEL:-lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct}
NUM_PROCESSES=${NUM_PROCESSES:-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29600}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/out/ov2_reference_${PROTOCOL}}

case "${SPLIT}" in
  short) TASK=ov2_videomme_short_wo_sutitle; OFFICIAL_F=128; MP=321489 ;;
  medium) TASK=ov2_videomme_medium_wo_sutitle; OFFICIAL_F=256; MP=136900 ;;
  long) TASK=ov2_videomme_long_wo_sutitle; OFFICIAL_F=640; MP=102400 ;;
  mlvu) TASK=mlvu_dev; OFFICIAL_F=512; MP=72900 ;;
  *) echo "unsupported SPLIT=${SPLIT}" >&2; exit 2 ;;
esac

case "${PROTOCOL}" in
  official) FRAME_CAP=${OFFICIAL_F}; FPS=1 ;;
  molmo384) FRAME_CAP=384; FPS=2 ;;
  *) echo "unsupported PROTOCOL=${PROTOCOL}" >&2; exit 2 ;;
esac

test -s "${MANIFEST}"
export LMMS_EVAL_DOC_ID_MANIFEST=${MANIFEST}
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false

output=${OUTPUT_ROOT}/${SPLIT}
mkdir -p "${output}"

python -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m lmms_eval \
  --model llava_onevision2 \
  --model_args "pretrained=${MODEL},trust_remote_code=True,attn_implementation=flash_attention_2,allow_all_kernels=True,messages_format=timestamp,fps=${FPS},max_num_frames=${FRAME_CAP},min_pixels=${MP},max_pixels=${MP},video_backend=frames" \
  --tasks "${TASK}" \
  --batch_size 1 \
  --log_samples \
  --output_path "${output}"
