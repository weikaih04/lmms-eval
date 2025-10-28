import random
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger


# ============================
# Visual Processing Functions
# ============================


def perspective_doc_to_visual(doc):
    """
    Return only the marked image (first frame with X mark).
    Does NOT include new_perspective image.
    
    Args:
        doc: Document containing marked_image and new_perspective
        
    Returns:
        List containing only the marked_image as PIL.Image
    """
    marked_image = doc.get("marked_image")
    
    if marked_image is not None:
        return [marked_image.convert("RGB")]
    else:
        # Fallback: return empty list if image is missing
        eval_logger.warning("marked_image is None in document")
        return []


# ============================
# Text Processing Functions
# ============================


def perspective_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Format the question text with choices.
    
    Args:
        doc: Document containing question and answer_choices
        lmms_eval_specific_kwargs: Optional kwargs for prompt customization
        
    Returns:
        Formatted question string
    """
    question = doc["question"].strip()
    
    # Parse answer choices
    # answer_choices is a string like '["Closer", "Further"]'
    answer_choices_str = doc.get("answer_choices", "")
    
    # Try to parse the choices
    try:
        import ast
        choices_list = ast.literal_eval(answer_choices_str)
        
        # Format as A) choice1, B) choice2
        formatted_choices = []
        for idx, choice in enumerate(choices_list):
            letter = chr(ord('A') + idx)  # A, B, C, ...
            formatted_choices.append(f"{letter}) {choice}")
        
        choices_text = "\n".join(formatted_choices)
        question = f"{question}\n{choices_text}"
    except:
        # If parsing fails, just append the raw string
        eval_logger.warning(f"Failed to parse answer_choices: {answer_choices_str}")
        question = f"{question}\n{answer_choices_str}"
    
    # Add post prompt if provided
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    
    return f"{question}{post_prompt}"


# ============================
# Result Processing Functions
# ============================


def perspective_process_results(doc, results):
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
    # For perspective QA, we have binary choices (A or B)
    all_choices = ["A", "B"]
    pred = parse_multi_choice_response(response, all_choices)
    
    # Get ground truth
    gt_ans = doc.get("answer", "").strip()
    
    # Calculate score
    score = 1.0 if pred == gt_ans else 0.0
    
    # Extract metadata for category-wise analysis
    question_type = doc.get("question_type", "unknown")
    
    # Infer split name from question_type if available
    # The splits are: distance_change_closer, distance_change_further, 
    # relative_position_left_left, relative_position_left_right, etc.
    split_name = "unknown"
    if "distance_change" in question_type:
        # Need to check the answer to determine closer/further
        try:
            import ast
            answer_choices_str = doc.get("answer_choices", "")
            choices_list = ast.literal_eval(answer_choices_str)
            
            # Map answer letter to actual choice
            answer_idx = ord(gt_ans) - ord('A')
            if 0 <= answer_idx < len(choices_list):
                actual_answer = choices_list[answer_idx].lower()
                if "closer" in actual_answer:
                    split_name = "distance_change_closer"
                elif "further" in actual_answer:
                    split_name = "distance_change_further"
        except:
            pass
    elif "relative_position" in question_type:
        # For relative position, we need more context
        # This will be filled from the actual split during evaluation
        split_name = question_type
    
    # Build accuracy dictionary with multiple breakdowns
    accuracy_dict = {
        "overall": score,
        f"question_type_{question_type}": score,
        f"split_{split_name}": score,
    }
    
    return {"accuracy": accuracy_dict}


# ============================
# Aggregation Functions
# ============================


def perspective_aggregate_results(results):
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
    eval_logger.info("AI2-THOR Perspective 411 Evaluation Results")
    eval_logger.info("=" * 80)
    eval_logger.info(f"Overall Accuracy: {overall_accuracy:.2f}% ({int(total_correct)}/{total_examples})")
    eval_logger.info("")
    
    # Group by category type
    question_types = {k: v for k, v in category_accuracy.items() if k.startswith("question_type_")}
    splits = {k: v for k, v in category_accuracy.items() if k.startswith("split_")}
    
    if question_types:
        eval_logger.info("By Question Type:")
        for category, acc in sorted(question_types.items()):
            cat_name = category.replace("question_type_", "")
            count = category_total[category]
            eval_logger.info(f"  {cat_name}: {acc:.2f}% ({int(category_correct[category])}/{count})")
        eval_logger.info("")
    
    if splits:
        eval_logger.info("By Split:")
        for category, acc in sorted(splits.items()):
            cat_name = category.replace("split_", "")
            count = category_total[category]
            eval_logger.info(f"  {cat_name}: {acc:.2f}% ({int(category_correct[category])}/{count})")
        eval_logger.info("")
    
    eval_logger.info("=" * 80)
    
    return round(overall_accuracy, 5)


# ============================
# Helper Functions
# ============================


def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter e.g., A, B.
    
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
    
    # Look for simple choices, e.g., A, B
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    
    # Look for choices with periods, e.g., A., B.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    
    # Look for choices with parentheses on the right, e.g., A)
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

