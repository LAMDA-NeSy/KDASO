import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import re
from collections import Counter

count = 0
error_rows = []


def extract_topic(text):
    """Extract content between <answer> and </answer> tags from the answer column."""
    raw_text = text
    if pd.isna(text) or text is None:
        return ""
    # Find the position of '</think>' to handle potential prefix noise
    topic_pos = text.lower().find('</think>')
    # Adjust slicing logic (note: if find returns -1, this might slice from end,
    # but we preserve original logic as requested without changing code behavior significantly)
    if topic_pos == -1:
        text_slice = text
    else:
        text_slice = text[topic_pos - 1:]

    # Match the pattern <answer>content</answer> (case-insensitive)
    match = re.search(r'<answer>(.+?)</answer>', str(text_slice), re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Increment global error counter and record the raw text for debugging
        global count
        count = count + 1
        global error_rows
        error_rows.append(raw_text)  # Record the erroneous text
        return ""


# Load the JSON dataset
# TODO: Replace 'INPUT_YOUR_OUTPUT_FILE_PATH' with your actual input file path
df = pd.read_json('INPUT_YOUR_OUTPUT_FILE_PATH', encoding='utf-8-sig')

# Apply the extraction function to the 'all_answer' column to generate predictions
df['prediction'] = df['all_answer'].apply(extract_topic)

# Extract ground truth labels and predictions into lists
labels = df['answer'].tolist()
predictions = df['prediction'].tolist()

# Normalize labels and predictions: convert to string, strip whitespace, and lowercase
labels = [str(label).strip().lower() for label in labels]
predictions = [str(pred).strip().lower() for pred in predictions]

# Create a new column to mark whether the prediction is correct (1: correct, 0: empty/invalid, 2: mismatch)
df['is_correct'] = 0
df['final_label'] = labels
df['final_prediction'] = predictions

exact_match = 0
error_indices = []

# Iterate through labels and predictions to evaluate performance
for idx, (label, prediction) in enumerate(zip(labels, predictions)):
    if prediction == '' or prediction is None:
        df.loc[idx, 'is_correct'] = 0
        continue
    if label == prediction:
        exact_match += 1
        df.loc[idx, 'is_correct'] = 1
    else:
        error_indices.append(idx)
        df.loc[idx, 'is_correct'] = 2

# Print basic statistics
correct_count = sum(1 for t, p in zip(labels, predictions) if t == p)
print(f"\nTotal samples: {len(labels)}")
print(f"Correct predictions: {correct_count}")
print(f"Incorrect predictions: {len(labels) - correct_count}")
print(f"Accuracy: {(correct_count/len(labels)):.4f}")