# AI2-THOR Perspective 411 Dataset

## Dataset Description

This task evaluates vision-language models on spatial reasoning and perspective-taking in 3D environments using the AI2-THOR simulator. The dataset contains 411 questions where models must reason about how objects appear from different viewpoints.

**Key Features:**
- **411 samples** with perspective-taking questions
- **2-choice binary questions** (A/B)
- **Single-image input**: Models see only the marked image, NOT the new perspective
- **Two question types**: Distance change and relative position

## Dataset Structure

**HuggingFace Dataset**: `weikaih/ai2thor-perspective-qa-annotated-411-splits`

**Fields:**
- `question`: The perspective question (e.g., "If I move to the 'X' marked point and turned left, will the object get closer or further?")
- `answer`: Ground truth answer (A or B)
- `answer_choices`: List of choices (e.g., ["Closer", "Further"])
- `question_type`: Type of question (perspective_distance_change or perspective_relative_position)
- `marked_image`: Image with X mark showing the target viewpoint
- `new_perspective`: Image from the new viewpoint (NOT used in evaluation)
- `scene_name`: AI2-THOR scene identifier
- `trajectory_id`: Trajectory identifier

## Splits

The dataset is divided into 6 splits based on question type and answer:

1. **distance_change_closer** (68 samples): Questions where objects get closer
2. **distance_change_further** (68 samples): Questions where objects get further
3. **relative_position_left_left** (68 samples): Left-to-left position changes
4. **relative_position_left_right** (70 samples): Left-to-right position changes
5. **relative_position_right_left** (69 samples): Right-to-left position changes
6. **relative_position_right_right** (68 samples): Right-to-right position changes

**Total**: 411 samples

## Evaluation

The model receives:
- ✅ **Marked image only** (showing the X mark for target viewpoint)
- ❌ **NO new perspective image**

This tests the model's ability to imagine and reason about spatial relationships from a different viewpoint without seeing the actual view.

## Metrics

- **Overall Accuracy**: Percentage of correct answers across all splits
- **By Question Type**: Accuracy for distance_change vs relative_position questions
- **By Split**: Accuracy for each of the 6 splits

## Usage

```bash
# Run evaluation with lmms-eval (all splits)
lmms-eval --model <model_name> \
          --tasks ai2thor_perspective_411 \
          --batch_size 1 \
          --output_path ./results/

# Run evaluation on specific split
lmms-eval --model <model_name> \
          --tasks ai2thor_perspective_411_distance_change_closer \
          --batch_size 1 \
          --output_path ./results/
```

## Question Types

### Distance Change Questions
Models must predict whether an object will appear closer or further from a new viewpoint.

**Example:**
```
Question: If I move to the 'X' marked point in the image and turned left, 
          will the tennisracket get closer or further away?
Choices: A) Closer, B) Further
```

### Relative Position Questions
Models must predict the relative position of an object from a new viewpoint.

**Example:**
```
Question: If I move to the 'X' marked point in the image and turned around, 
          will the apple be on my left or right?
Choices: A) Left, B) Right
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{ai2thor_perspective_411,
  title={AI2-THOR Perspective-Taking Dataset},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/datasets/weikaih/ai2thor-perspective-qa-annotated-411-splits}
}
```

