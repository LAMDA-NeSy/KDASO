import pandas as pd
import json
import re
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np

count = 0
error_rows = []


# 2. Iterate through each row to extract content after [topic]
def extract_topic(text):
    """Extract content between [topic] and <eoa> from the answer column."""
    raw_text = text
    if pd.isna(text) or text is None:
        return ""
    topic_pos = text.lower().find('</think>')
    # Note: If find returns -1, this slice might behave unexpectedly,
    # but we preserve the original logic as requested.
    text = text[topic_pos - 1:]

    # Match pattern [topic]content<eoa> (case-insensitive)
    match = re.search(r'\[topic\](.+?)<eoa>', str(text), re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Fallback 1: Match content inside square brackets [...]
        match_new = re.search(r'\[(.+?)\]', str(text), re.IGNORECASE)
        if match_new:
            return match_new.group(1).strip()
        else:
            # Fallback 2: Match content starting with [ and ending with >
            match_new_new = re.search(r'\[(.+?)>', str(text), re.IGNORECASE)
            if match_new_new:
                return match_new_new.group(1).strip()
            else:
                # Extraction failed: increment error counter and log details
                global count
                count = count + 1
                print(count, raw_text)
                global error_rows
                error_rows.append(raw_text)  # Record the erroneous text
                return ""


def manual_accuracy(y_true, y_pred):
    """
    Calculate accuracy with a relaxed matching rule.
    Matches if strings are equal or one contains the other.
    Skips predictions that are None or too short (<4 chars).
    """
    correct_count = 0
    total_count = len(y_true)

    for true_val, pred_val in zip(y_true, y_pred):
        true_val = true_val.lower()
        pred_val = pred_val.lower()

        # Skip invalid predictions
        if pred_val == None or len(pred_val) < 4:
            continue
        # Relaxed matching: exact match or substring match
        elif true_val == pred_val or true_val in pred_val or pred_val in true_val:
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy


def balanced_accuracy_score_manual(y_true, y_pred):
    """
    Manually calculate multi-class balanced accuracy for string category labels.

    Args:
        y_true (pd.Series or array-like): Ground truth labels (string categories)
        y_pred (pd.Series or array-like): Predicted labels (string categories)

    Returns:
        float: Balanced accuracy score
    """
    # Convert to numpy array for easier manipulation (compatible with pandas Series)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure lengths match
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    # Get all unique classes present in ground truth
    classes = np.unique(y_true)
    print(classes)

    recalls = []
    for cls in classes:
        actual_count = 0
        correct_count = 0

        # Iterate through each sample
        for true_val, pred_val in zip(y_true, y_pred):
            if true_val == cls:
                true_val = true_val.lower()
                pred_val = pred_val.lower()
                actual_count += 1
                # Apply relaxed matching rule
                if (true_val == pred_val) or (true_val in pred_val) or (pred_val in true_val):
                    correct_count += 1

        # Calculate recall for this class
        recall = correct_count / actual_count if actual_count > 0 else 0.0
        recalls.append(recall)

    # Return the average recall across all classes
    return np.mean(recalls)


# 3. Create the prediction column
# TODO: Replace 'INPUT_YOUR_OUTPUT_FILE_PATH' with your actual file path
df = pd.read_json('INPUT_YOUR_OUTPUT_FILE_PATH', encoding='utf-8-sig')

df['prediction'] = df['all_answer'].apply(extract_topic)

# Calculate metrics using custom functions
accuracy = manual_accuracy(df['answer'], df['prediction'])
balanced_accuracy = balanced_accuracy_score_manual(df['answer'], df['prediction'])

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")