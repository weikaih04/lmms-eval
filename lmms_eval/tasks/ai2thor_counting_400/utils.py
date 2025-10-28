import random
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger


# ============================
# Visual Processing Functions
# ============================


def counting_doc_to_visual(doc):
    """
    Return all available agent frames (3-5 frames depending on the question).
    Does NOT include topdown_map.
    
    Args:
        doc: Document containing frame_0 to frame_4 and topdown_map
        
    Returns:
        List of PIL.Image objects (only agent frames, no topdown map)
    """
    frames = []
    
    # Iterate through all possible frames (0-4)
    for i in range(5):
        frame = doc.get(f'frame_{i}')
        if frame is not None:
            frames.append(frame.convert("RGB"))
    
    return frames


# ============================
# Text Processing Functions
# ============================


def counting_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Format the question text with choices.

    Args:
        doc: Document containing question and choices
        lmms_eval_specific_kwargs: Optional kwargs for prompt customization

    Returns:
        Formatted question string
    """
    question = doc["question"].strip()

    # Format choices - choices already contain "A) ...", "B) ..." format
    choices = doc.get("choices", [])
    if choices:
        choices_text = "\n".join(choices)
        full_question = f"{question}\n{choices_text}"
    else:
        full_question = question

    # Add post prompt if provided
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{full_question}{post_prompt}"


# ============================
# Result Processing Functions
# ============================


def counting_process_results(doc, results):
    """
    Process the model's response and compare with ground truth.
    
    Args:
        doc: Document containing answer and metadata
        results: List of model predictions
        
    Returns:
        Dictionary with accuracy metrics broken down by categories
    """
    response = results[0].strip() if results else ""
    
    # Parse the response to extract the answer choice
    all_choices = ["A", "B", "C", "D"]
    pred = parse_multi_choice_response(response, all_choices)
    
    # Get ground truth
    gt_ans = doc.get("answer", "").strip()
    
    # Calculate score
    score = 1.0 if pred == gt_ans else 0.0
    
    # Extract metadata for category-wise analysis
    question_type = doc.get("question_type", "unknown")
    difficulty = doc.get("difficulty", "unknown")
    movement_type = doc.get("movement_type", "unknown")
    total_frames = doc.get("total_frames", 0)
    
    # Build accuracy dictionary with multiple breakdowns
    accuracy_dict = {
        "overall": score,
        f"question_type_{question_type}": score,
        f"difficulty_{difficulty}": score,
        f"movement_{movement_type}": score,
        f"frames_{total_frames}": score,
    }
    
    return {"accuracy": accuracy_dict}


# ============================
# Aggregation Functions
# ============================


def counting_aggregate_results(results):
    """
    Aggregate results across all samples with detailed breakdowns.
    
    Args:
        results: List of accuracy dictionaries from process_results
        
    Returns:
        Overall accuracy as a percentage
    """
    total_correct = 0
    total_examples = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    for result in results:
        # Overall accuracy
        total_correct += result["overall"]
        total_examples += 1
        
        # Category-wise accuracy
        for category, score in result.items():
            if category != "overall":
                category_correct[category] += score
                category_total[category] += 1
    
    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_examples) * 100 if total_examples > 0 else 0.0
    
    # Calculate category-wise accuracy
    category_accuracy = {}
    for category in category_correct:
        if category_total[category] > 0:
            category_accuracy[category] = (category_correct[category] / category_total[category]) * 100
    
    # Print detailed results
    eval_logger.info("=" * 80)
    eval_logger.info("AI2-THOR Counting 400 Evaluation Results")
    eval_logger.info("=" * 80)
    eval_logger.info(f"Overall Accuracy: {overall_accuracy:.2f}% ({int(total_correct)}/{total_examples})")
    eval_logger.info("")
    
    # Group by category type
    question_types = {k: v for k, v in category_accuracy.items() if k.startswith("question_type_")}
    difficulties = {k: v for k, v in category_accuracy.items() if k.startswith("difficulty_")}
    movements = {k: v for k, v in category_accuracy.items() if k.startswith("movement_")}
    frames = {k: v for k, v in category_accuracy.items() if k.startswith("frames_")}
    
    if question_types:
        eval_logger.info("By Question Type:")
        for category, acc in sorted(question_types.items()):
            cat_name = category.replace("question_type_", "")
            count = category_total[category]
            eval_logger.info(f"  {cat_name}: {acc:.2f}% ({int(category_correct[category])}/{count})")
        eval_logger.info("")
    
    if difficulties:
        eval_logger.info("By Difficulty:")
        for category, acc in sorted(difficulties.items()):
            cat_name = category.replace("difficulty_", "")
            count = category_total[category]
            eval_logger.info(f"  {cat_name}: {acc:.2f}% ({int(category_correct[category])}/{count})")
        eval_logger.info("")
    
    if movements:
        eval_logger.info("By Movement Type:")
        for category, acc in sorted(movements.items()):
            cat_name = category.replace("movement_", "")
            count = category_total[category]
            eval_logger.info(f"  {cat_name}: {acc:.2f}% ({int(category_correct[category])}/{count})")
        eval_logger.info("")
    
    if frames:
        eval_logger.info("By Number of Frames:")
        for category, acc in sorted(frames.items()):
            cat_name = category.replace("frames_", "")
            count = category_total[category]
            eval_logger.info(f"  {cat_name} frames: {acc:.2f}% ({int(category_correct[category])}/{count})")
        eval_logger.info("")
    
    eval_logger.info("=" * 80)
    
    return round(overall_accuracy, 5)


# ============================
# Helper Functions
# ============================


def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter e.g., A, B, C, D.
    
    Adapted from MMMU's utils.py.
    
    Args:
        response: Model's generated response
        all_choices: List of valid choice letters
        
    Returns:
        Predicted choice letter
    """
    # Clean response of unwanted characters
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match
    
    candidates = []
    
    # Look for choices with parentheses, e.g., (A)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    
    # Look for simple choices, e.g., A, B, C
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    
    # Look for choices with periods, e.g., A., B., C.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # Look for choices with parentheses on the right, e.g., A), B), C)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice})" in response:
                candidates.append(choice)

    # If no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # If more than one candidate, choose the last one found
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]
    
    return pred_index

