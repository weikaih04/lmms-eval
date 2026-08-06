#!/usr/bin/env bash
set -euo pipefail

: "${MOLMO2_CODEC_REPO:=/fsx/home/weikai.huang/molmo2_codec/mm_olmo}"
: "${MOLMO2_CODEC_PTOK:?set MOLMO2_CODEC_PTOK to the Stage-1 checkpoint}"
: "${MOLMO2_CODEC_STAGE2:?set MOLMO2_CODEC_STAGE2 to a consolidated Stage-2 checkpoint}"
: "${MOLMO2_CODEC_GAMMA:?set MOLMO2_CODEC_GAMMA to the fixed gamma artifact}"
: "${MOLMO2_CODEC_CACHE:=./out/molmo2_codec_cache_8192}"
: "${MOLMO2_CODEC_GOP_READ_DIRS:=${MOLMO2_CODEC_CACHE}/gop_selected}"
: "${MOLMO2_CODEC_MOTION_READ_DIRS:=${MOLMO2_CODEC_CACHE}/motion}"
: "${TASK:=molmo2_mlvu_dev}"
: "${LIMIT:=1}"
: "${OUTPUT_PATH:=./out/molmo2_codec_smoke}"
: "${MOLMO2_CODEC_P_VARIANT:=real}"
: "${TRACE_OUTPUT:=${OUTPUT_PATH}/codec_trace.jsonl}"
: "${VIDEO_BACKEND:=codec}"
: "${NUM_PROCESSES:=1}"
: "${MAIN_PROCESS_PORT:=29500}"

mkdir -p \
  "${MOLMO2_CODEC_CACHE}/gop_selected" \
  "${MOLMO2_CODEC_CACHE}/motion" \
  "${MOLMO2_CODEC_CACHE}/features"

launcher=(python -m lmms_eval)
if (( NUM_PROCESSES > 1 )); then
  launcher=(accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}" -m lmms_eval)
fi

"${launcher[@]}" \
  --model molmo2_codec \
  --model_args "training_repo=${MOLMO2_CODEC_REPO},pretrained=${MOLMO2_CODEC_STAGE2},ptok_checkpoint=${MOLMO2_CODEC_PTOK},gamma_artifact=${MOLMO2_CODEC_GAMMA},video_backend=${VIDEO_BACKEND},p_variant=${MOLMO2_CODEC_P_VARIANT},visual_token_budget=8192,timeline_max_frames=2048,max_frames=101,seq_len=16384,gop_cache_dir=${MOLMO2_CODEC_CACHE}/gop_selected,gop_cache_read_dirs=${MOLMO2_CODEC_GOP_READ_DIRS},motion_cache_dir=${MOLMO2_CODEC_CACHE}/motion,motion_cache_read_dirs=${MOLMO2_CODEC_MOTION_READ_DIRS},feature_cache_dir=${MOLMO2_CODEC_CACHE}/features,trace_output=${TRACE_OUTPUT}" \
  --tasks "${TASK}" \
  --batch_size 1 \
  --limit "${LIMIT}" \
  --log_samples \
  --output_path "${OUTPUT_PATH}"
