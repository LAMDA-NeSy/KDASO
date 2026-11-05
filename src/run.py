import pandas as pd
import json
import os
import re
import requests
from src.setting.config import *
from src.setting.prompts import *


def load_test_data(file_path, info_column, question_column):
    """Loading test data and extracting specific info column"""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    processed_data = []
    for obj in data:
        processed_data.append({
            "full_record": obj,
            "input_text": obj.get(info_column, ""),
            "question": obj.get(question_column, ""),
        })

    return processed_data


def call_llm(user_prompt, system_prompt, params=LLM_PARAMS):
    """Generate guidance by LLMs"""
    headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
    data = {
        "model": LLM_PATH,
        "messages": [
            {"role": "system", "content": REFINE_PROMPT_1 + system_prompt + REFINE_PROMPT_2},
            {"role": "user", "content": user_prompt}
        ],
        **params
    }
    try:
        response = requests.post(LLM_URL, headers=headers, json=data, timeout=180)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"⚠️ LLM Reject: {e}")
        return None


def extract_guidance(llm_output):
    """Extract guidance from tags"""
    if not llm_output:
        return None
    # Try to find <advice>...</advice>
    match = re.search(r'<advice>(.*?)</advice>', llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if no tag, maybe return the whole output or None depending on needs
    return llm_output.strip()


def get_system_prompt(experiment_name, row_index):
    df = pd.read_csv(f"./result/{experiment_name}/rule_retriever/predictions.csv")
    with open(f"./result/{experiment_name}/generate_rule.json", "r",
              encoding="utf-8-sig") as f:
        json_data = json.load(f)

    row = df.iloc[row_index]
    matched_columns = row[row == 1].index.tolist()

    matched_rules = {}
    missing_keys = []

    for col_name in matched_columns:
        if col_name in json_data:
            matched_rules[col_name] = json_data[col_name]
        else:
            missing_keys.append(col_name)

    if missing_keys:
        print(f"⚠️ Not Found：{missing_keys}")

    return matched_rules


def main(experiment_name, dataset_name):
    # ===== Config =====
    TEST_JSON = f"./data/{dataset_name}/test.json"
    OUTPUT_JSON = f"./result/{experiment_name}/final_output.json"

    # Load Info Config
    config_path = f"./data/{dataset_name}/info.json"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8-sig") as f:
        config_data = json.load(f)
        INFO_COLUMN = config_data["INFO_COLUMN"]
        QUESTION_COLUMN = config_data["QUESTION_COLUMN"]

    print("📦 Loading test data...")
    test_data_list = load_test_data(TEST_JSON, INFO_COLUMN, QUESTION_COLUMN)

    results = []
    total_cases = len(test_data_list)

    print(f"🚀 Processing {total_cases} test cases with LLM...")

    for i, item in enumerate(test_data_list):
        full_record = item["full_record"]
        input_text = item["input_text"]
        question = item["question"]

        # Construct User Prompt using only INFO_COLUMN content
        # You can adjust the template below as needed
        user_prompt = f"{input_text}\n\n{question}"
        system_prompt = str(get_system_prompt(experiment_name, i))

        print(f"[{i + 1}/{total_cases}] Calling LLM...")

        # Call LLM with System Prompt and constructed User Prompt
        llm_output = call_llm(user_prompt, system_prompt)

        # Construct result object
        result_entry = {
            "original_content": full_record,
            "user_prompt": user_prompt,
            "llm_raw_output": llm_output
        }

        results.append(result_entry)

    # Save final results
    print(f"💾 Saving final results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8-sig") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Processed {len(results)} cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KDARO")
    parser.add_argument("--experiment_name", type=str, default="exp_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="Experiment name")
    parser.add_argument("--dataset_name", type=str, default="training_set", help="Dataset name")

    main(experiment_name, dataset_name)
