import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import re
import numpy as np

# Counter for tracking extraction failures
count = 0
# List to store raw text rows where extraction failed
error_rows = []


def extract_topic(text):
    """
    Extract content between <answer> and </answer> tags from the text.
    Returns the extracted content stripped of whitespace, or "-1" if extraction fails.
    """
    raw_text = text

    # Handle null or NaN values
    if pd.isna(text) or text is None:
        return ""

    # Locate the end of the thinking process (</think>) to focus on the final answer
    # Note: The original logic searches for '</think>' and slices from slightly before it
    topic_pos = text.lower().find('</think>')

    # Safety check in case '</think>' is not found (find returns -1)
    if topic_pos == -1:
        topic_pos = 0
    else:
        # Adjust slice start position as per original logic
        topic_pos = max(0, topic_pos - 1)

    text = text[topic_pos:]

    # Match the pattern <answer>content</answer> (case-insensitive)
    match = re.search(r'<answer>(.+?)</answer>', str(text), re.IGNORECASE)

    if match:
        return match.group(1).strip()
    else:
        # Increment global error counter and record the raw text for debugging
        global count
        count += 1
        global error_rows
        error_rows.append(raw_text)
        return "-1"


# Load the JSON dataset
# TODO: Replace 'INPUT_YOUR_OUTPUT_FILE_PATH' with your actual input file path
df = pd.read_json('INPUT_YOUR_OUTPUT_FILE_PATH', encoding='utf-8-sig')

# Apply the extraction function to the 'all_answer' column to create 'prediction'
df['prediction'] = df['all_answer'].apply(extract_topic)

# Map option letters (a, b, c, d) to integers (0, 1, 2, 3)
# Maps "-1" (extraction failure) to 0 as a default fallback
option_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, "-1": 0}
df['prediction'] = df['prediction'].map(option_mapping)

# Ensure the 'prediction' column is of integer type
df['prediction'] = df['prediction'].astype(int)

# Convert the prediction series to a numpy array
predictions = df['prediction'].values

# Save the predictions to a .npy file
# TODO: Replace 'YOUR_OUTPUT_FILE.npy' with your desired output filename
np.save('YOUR_OUTPUT_FILE.npy', predictions)

print(f"Generate .npy file successfully.")