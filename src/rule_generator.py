import json
import numpy as np
import re
import requests
from sentence_transformers import SentenceTransformer
from src.setting.config import *
from src.setting.prompts import *

def load_data(file_path, SEARCH_LIST):
    """Loading data"""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    texts, samples_info = [], []
    for obj in data:

        values = []
        for col in SEARCH_LIST:
            val = obj.get(col, "")
            if val:  # Only add if not empty/None
                values.append(str(val).strip())

        # Join with separator if there are values, otherwise use default text
        text = " | ".join(values) if values else "No content"
        texts.append(text)

        base_info = {}
        for key, value in obj.items():
            base_info[key] = value
        samples_info.append(base_info)

    return data, texts, samples_info


def call_llm(user_prompt, system_prompt=GENERATE_PROMPT, params=LLM_PARAMS):
    """Generate guidance by LLMs"""
    headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
    data = {
        "model": LLM_PATH,
        "messages": [
            {"role": "system", "content": system_prompt},
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


def build_user_prompt(current_case, retrieved_cases):
    """Build user prompt"""
    prompt = f"### Information for the Current Case Requiring Adjudication:\n{current_case}\n\n"
    prompt += "### Information on Similar Cases:\n"
    for idx, case in enumerate(retrieved_cases, 1):
        prompt += f"#### Similar Case #{idx}:\n{json.dumps(case, ensure_ascii=False, indent=2)}\n\n"
    return prompt


def extract_guidance(llm_output):
    """Extract guidance"""
    if not llm_output:
        return None
    match = re.search(r'<advice>(.*?)</advice>', llm_output, re.DOTALL)
    return match.group(1).strip() if match else None


def main(experiment_name, dataset_name, top_k=4):
    # ===== config =====
    ### ===== Data config =====
    TRAIN_JSON = f"./data/{dataset_name}/train.json"
    TEST_JSON = f"./data/{dataset_name}/test.json"
    OUTPUT_JSON = f"./result/{experiment_name}/generate_guidance.json"

    ### ===== Model config =====
    with open(f"./data/{dataset_name}/info.json", "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        SEARCH_LIST = data["SEARCH_LIST"]
        INFO_COLUMN = data["INFO_COLUMN"]
        ANSWER_COLUMN = data["ANSWER_COLUMN"]

    print("📦 Loading train & test data...")
    _, train_texts, train_info = load_data(TRAIN_JSON, SEARCH_LIST)
    _, test_texts, test_info = load_data(TEST_JSON, SEARCH_LIST)

    # Encode
    print("🔍 Encoding with BGE...")
    model = SentenceTransformer(MODEL_NAME)
    train_emb = model.encode(train_texts, show_progress_bar=True, normalize_embeddings=True)
    test_emb = model.encode(test_texts, show_progress_bar=True, normalize_embeddings=True)

    results = []
    print(f"🚀 Processing {len(test_info)} test cases...")

    for i in range(len(test_info)):
        info_i = test_info[i]

        # 1️⃣ Search for similar cases (cosine similarity)
        sims = np.dot(train_emb, test_emb[i])
        top_indices = np.argsort(sims)[::-1][:min(top_k, len(train_info))]

        retrieved = []
        for j in top_indices:
            info_j = train_info[j]

            plan = {}
            for col in SEARCH_LIST:
                val = info_j[col]
                plan[col] = val
            plan["Answer"] = info_j[ANSWER_COLUMN]

            retrieved.append({
                "Info": info_j[INFO_COLUMN],
                "Chain of Thought": plan,
            })

        # 2️⃣ Call upon an LLM to generate guidance
        print(f"[{i + 1}/{len(test_info)}] Generating guidance for the case...")
        current = {"info": info_i[INFO_COLUMN]}
        user_prompt = build_user_prompt(current, retrieved)
        llm_output = call_llm(user_prompt)
        guidance = extract_guidance(llm_output)

        # 3️⃣ Construct an output object
        base_info = {}
        for key, value in info_i.items():
            base_info[key] = value

        results.append({
            "original_content": base_info,
            "retrieved_cases": retrieved,
            "generated_guidance": guidance,
            "llm_raw_output": llm_output  # 保留原始输出便于调试
        })

    # 4️⃣ Save as a single JSON file
    print(f"💾 Saving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8-sig") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Processed {len(results)} cases.")

    rule_file_path = f"./data/{dataset_name}/rule.json"

    with open(rule_file_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)

    max_index = -1
    for key in rules.keys():
        if key.startswith('R'):
            try:
                num = int(key[1:])
                if num > max_index:
                    max_index = num
            except ValueError:
                continue

    next_index = max_index + 1
    new_rules_count = 0

    for item in results:
        llm_output = item.get('llm_raw_output', '')

        index = llm_output.find('</think>')
        if index != -1:
            llm_output = llm_output[index:]

        match = re.search(r'<advice>(.*?)</advice>', llm_output, re.DOTALL)

        if match:
            advice_content = match.group(1).strip()

            if not advice_content:
                continue

            new_key = f"R{next_index}"

            rules[new_key] = advice_content

            next_index += 1
            new_rules_count += 1
        else:
            print("Warning: No <advice> tag found in one of the llm_raw_output entries.")

    with open(f"./result/{experiment_name}/generate_rule.json", 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    print(f"\nSuccess! Added {new_rules_count} new rules to ./result/{experiment_name}/generate_rule.json.")
    print(f"Total rules now: {len(rules)}")
