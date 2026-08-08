# Molmo2 AdaCodec paired evaluation

This directory contains the reproducible evaluation entry points used for the
Molmo2 AdaCodec V6 benchmark.  They intentionally separate two protocols:

- `official`: reproduce each model's published/native evaluation settings.
- `molmo384`: compare on the same frozen question IDs and the Molmo2 paper's
  2 FPS, full-span, maximum-384-frame sampling policy.

The frozen question selection is stored in
`../llava_onevision2_repro/manifests/VIDEO_QUALIFICATION_RANDOM200_V2_MANIFEST.json`.
It contains exactly 200 task-native document IDs for Video-MME Short, Medium,
Long, and MLVU.

## Molmo2 Dense/Codec pair

Set the three checkpoint paths and run one split:

```bash
export MOLMO2_CODEC_REPO=/path/to/molmo2-codec/mm_olmo
export MOLMO2_CODEC_PTOK=/path/to/ptokenizer.pt
export MOLMO2_CODEC_STAGE2=/path/to/stage2-consolidated.pt
export MOLMO2_CODEC_GAMMA=/path/to/gamma.json

SPLIT=long PROTOCOL=molmo384 \
  bash examples/molmo2_codec/run_paired_video.sh
```

The launcher evaluates Dense and Codec with the same checkpoint, task-native
document IDs, prompt, source-frame cap, and decoding settings.  Codec is capped
at 8,192 visual patch tokens.

## Released LLaVA-OneVision-2 reference

```bash
# Published benchmark-specific setting.
SPLIT=long PROTOCOL=official \
  bash examples/molmo2_codec/run_ov2_reference.sh

# Molmo2 384-frame control setting.
SPLIT=long PROTOCOL=molmo384 \
  bash examples/molmo2_codec/run_ov2_reference.sh
```

`official` uses the LLaVA-OneVision-2 reproduction guide's per-task frame and
pixel settings.  `molmo384` lowers/raises only the temporal policy to 2 FPS and
at most 384 source frames while retaining OV2's task-specific spatial setting.

## Alignment boundary

The current controlled protocol freezes exact QA IDs, video paths, FPS policy,
full-span coverage, and maximum frame count.  It does **not** claim bit-exact
source-frame indices across model families: Molmo2 and OV2 use independent
video loaders whose final linspace rounding differs.  Strict pixel-identical
cross-model evaluation requires a separately frozen timestamp/frame-index
artifact consumed by both adapters.
