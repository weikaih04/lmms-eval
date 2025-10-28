# AI2-THOR Counting 400 Dataset

## Dataset Description

This task evaluates vision-language models on object counting in 3D environments using the AI2-THOR simulator. The dataset contains 400 questions where models must count objects visible across multiple first-person view frames.

**Key Features:**
- **400 samples** with object counting questions
- **3-5 frames per question** (variable, depending on trajectory length)
- **4-choice multiple choice** (A/B/C/D)
- **Multi-image input**: Models see only agent frames, NOT the top-down map

## Dataset Structure

**HuggingFace Dataset**: `weikaih/ai2thor-counting-final-400`

**Fields:**
- `question`: The counting question (e.g., "Count the number of baseballbat(s) present.")
- `answer`: Ground truth answer (A/B/C/D)
- `choices`: List of formatted choices (e.g., ["A) 4", "B) 1", "C) 2", "D) 3"])
- `frame_0` to `frame_4`: Agent perspective images (some may be None)
- `topdown_map`: Top-down view (NOT used in evaluation)
- `total_frames`: Number of available frames (3-5)
- `question_type`: Type of question
- `difficulty`: Difficulty level
- `movement_type`: Type of agent movement
- `scene_name`: AI2-THOR scene identifier

## Frame Distribution

- **3 frames**: 90 samples (22.5%)
- **4 frames**: 131 samples (32.8%)
- **5 frames**: 179 samples (44.8%)

## Evaluation

The model receives:
- ✅ **Agent frames only** (frame_0 to frame_4, depending on total_frames)
- ❌ **NO top-down map**

This tests the model's ability to count objects from first-person perspectives without global scene understanding.

## Metrics

- **Overall Accuracy**: Percentage of correct answers
- **By Question Type**: Accuracy broken down by question type
- **By Difficulty**: Accuracy broken down by difficulty level
- **By Movement Type**: Accuracy broken down by movement type
- **By Frame Count**: Accuracy for 3-frame, 4-frame, and 5-frame questions

## Usage

```bash
# Run evaluation with lmms-eval
lmms-eval --model <model_name> \
          --tasks ai2thor_counting_400 \
          --batch_size 1 \
          --output_path ./results/
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{ai2thor_counting_400,
  title={AI2-THOR Counting Dataset},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/datasets/weikaih/ai2thor-counting-final-400}
}
```

