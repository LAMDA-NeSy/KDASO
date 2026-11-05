import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from lightgbm import LGBMClassifier
import csv
import logging
import os
from sklearn.cluster import KMeans
import math
import joblib
import json
from src.setting.config import *
import shutil


def create_cluster_centers_samples(df, label_1_count, feature_columns=None):
    """
    Create representative samples based on clustering centers
    """
    n_clusters = max(int(label_1_count), 50)
    print(f"n_clusters set to: {n_clusters}")
    label_0_data = df[df['rule_new'] == 0]
    label_0_data = label_0_data.copy()
    label_0_data.columns = label_0_data.columns.astype(str)
    print(f"Negative samples (label_0) count: {label_0_data.shape[0]}")

    if len(label_0_data) < n_clusters:
        print(
            f"Warning: insufficient negative samples, using random sampling (target {n_clusters}, available {len(label_0_data)})")
        return label_0_data.sample(n=min(n_clusters, len(label_0_data)), random_state=42)

    numeric_cols = label_0_data.select_dtypes(include=['int64', 'float64']).columns
    label_0_data[numeric_cols] = label_0_data[numeric_cols].fillna(label_0_data[numeric_cols].mean())

    string_cols = label_0_data.columns.difference(numeric_cols)
    for col in string_cols:
        label_0_data[col] = label_0_data[col].fillna(label_0_data[col].mode()[0])

    nan_columns = label_0_data.columns[label_0_data.isna().any()].tolist()
    if nan_columns:
        print(f"Columns with NaN values (dropped): {nan_columns}")
    label_0_data = label_0_data.drop(columns=nan_columns)

    if feature_columns is None:
        feature_columns = label_0_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Automatically selected clustering feature columns: {feature_columns}")

    if not feature_columns:
        print("Warning: no suitable numeric feature columns found, using random sampling")
        return label_0_data.sample(n=n_clusters, random_state=42)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(label_0_data[feature_columns])

    cluster_centers = kmeans.cluster_centers_
    synthetic_samples = []
    for i, center in enumerate(cluster_centers):
        new_sample = label_0_data.iloc[0].copy()
        for j, col in enumerate(feature_columns):
            if j < len(center):
                new_sample[col] = center[j]

        non_numeric_cols = [col for col in label_0_data.columns
                            if col not in feature_columns and col != 'rule_new']
        for col in non_numeric_cols:
            cluster_samples = label_0_data[kmeans.labels_ == i]
            if len(cluster_samples) > 0:
                most_common = cluster_samples[col].mode()
                if len(most_common) > 0:
                    new_sample[col] = most_common[0]

        synthetic_samples.append(new_sample)

    print(f"Number of synthetic negative samples generated from cluster centers: {len(synthetic_samples)}")
    return pd.DataFrame(synthetic_samples)

def main(experiment_name, dataset_name, use_clustering, true_sample_ratio):

    train_data = f"./data/{dataset_name}/train.json"
    test_data = f"./data/{dataset_name}/test.json"
    OUTPUT = f"./result/{experiment_name}/rule_retriever"
    os.makedirs(f"{OUTPUT}", exist_ok=True)
    os.makedirs(f"{OUTPUT}/model", exist_ok=True)

    with open(f"./data/{dataset_name}/info.json", "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        INFO_COLUMN = data["INFO_COLUMN"]
        RULE_COLUMN = data["RULE_COLUMN"]

    rule_path = f"./result/{experiment_name}/rule_refiner"
    all_items = os.listdir(rule_path)
    step_numbers = []
    max_num = 0
    for item in all_items:
        if item.startswith("step_"):
            try:
                num = int(item.split("_")[1])
                step_numbers.append(num)
            except (IndexError, ValueError):
                continue
    if step_numbers:
        max_num = max(step_numbers)
        print(f"The max step is: {max_num}")
    else:
        print("Error")

    file_1 = rule_path + f"/step_{max_num}/experiences.json"
    file_2 = f"./data/{dataset_name}/rule.json"
    merged_data = []
    if os.path.exists(file_1):
        with open(file_1, 'r', encoding='utf-8-sig') as f:
            content_1 = json.load(f)
            if isinstance(content_1, list):
                merged_data.extend(content_1)
            elif isinstance(content_1, dict):
                merged_data.append(content_1)

    # 处理第二个文件
    if os.path.exists(file_2):
        with open(file_2, 'r', encoding='utf-8-sig') as f:
            content_2 = json.load(f)
            if isinstance(content_2, list):
                merged_data.extend(content_2)
            elif isinstance(content_2, dict):
                merged_data.append(content_2)

    rule_path = f"./result/{experiment_name}/final_rule_base.json"
    with open(rule_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"合并完成，共 {len(merged_data)} 条数据，已保存至: {file_2}")

    with open(file_2, "r", encoding="utf-8-sig") as f:
        json_data = json.load(f)
        rules = list(json_data.keys())

    # Initialize answer file with header
    answer_file = f"{OUTPUT}/answer.csv"
    with open(answer_file, mode="w", newline="", encoding="utf-8-sig") as scene_file:
        writer = csv.writer(scene_file)
        writer.writerow(["Rule-Name",
                         "XGB-Precision", "XGB-Recall"])
    predict_result = f"{OUTPUT}/predictions.csv"
    input_df = pd.read_json(test_data, encoding='utf-8-sig', dtype=str)
    output_df = pd.DataFrame()
    output_df['index'] = range(len(input_df))
    output_df.to_csv(predict_result, index=False)

    count = 0

    for rule in rules:
        count += 1
        print(f"========== Processing scenario {count}/{len(rules)}: {rule} ==========")

        df = pd.read_json(train_data, encoding='utf-8-sig', dtype=str)

        df['rule_new'] = 0
        df.loc[df[RULE_COLUMN].str.contains(rule, na=False), 'rule_new'] = 1

        label_1_count = df[df['rule_new'] == 1].shape[0]

        if use_clustering:
            # Method 1: Clustering-based sampling (currently disabled)
            label_0_sample = create_cluster_centers_samples(df.copy(), int(true_sample_ratio*label_1_count)+1)
        else:
            # Method 2: Random sampling
            label_0_sample = df[df['rule_new'] == 0].sample(n=int(true_sample_ratio * math.ceil(label_1_count)) + 1, random_state=42)

        df = pd.concat([df[df['rule_new'] == 1], label_0_sample])
        df = df.sample(frac=1).reset_index(drop=True)

        # Feature separation
        model = SentenceTransformer(MODEL_NAME)

        if isinstance(INFO_COLUMN, list) and len(INFO_COLUMN) > 1:
            df["combined_text"] = df[INFO_COLUMN].fillna("").astype(str).apply(lambda x: " ".join(x), axis=1)
        else:
            df["combined_text"] = df[INFO_COLUMN].fillna("").astype(str)

        embeddings = model.encode(df["combined_text"].tolist())

        X_train_merge = embeddings.copy()
        y_train_merge = df["rule_new"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_merge, y_train_merge,
            test_size=0.4, stratify=y_train_merge, random_state=42
        )
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

        # Test set processing
        test_df = pd.read_json(test_data, encoding='utf-8-sig', dtype=str)
        if isinstance(INFO_COLUMN, list) and len(INFO_COLUMN) > 1:
            test_df["combined_text"] = test_df[INFO_COLUMN].fillna("").astype(str).apply(lambda x: " ".join(x), axis=1)
        else:
            test_df["combined_text"] = test_df[INFO_COLUMN].fillna("").astype(str)
        test_embeddings = model.encode(test_df["combined_text"].tolist())

        X_test = test_embeddings.copy()

        # Model training
        print("Start training XGB...")
        clf_xgb = XGBClassifier()
        clf_xgb.fit(X_train, y_train)
        y_pred_val = clf_xgb.predict(X_val)
        precision_xgb = precision_score(y_val, y_pred_val, zero_division=0)
        recall_xgb = recall_score(y_val, y_pred_val, zero_division=0)
        print(f"XGB validation recall: {recall_xgb:.4f}")

        # Save results to CSV
        with open(answer_file, mode="a", newline="", encoding="utf-8") as rule_file:
            writer = csv.writer(rule_file)
            writer.writerow([rule, precision_xgb, recall_xgb])

        # Test set prediction
        y_pred = clf_xgb.predict(X_test)
        print(f"Test set prediction completed, predicted positive count: {sum(y_pred)}")

        # Save prediction results
        input_df = pd.read_csv(predict_result)
        input_df[f'{rule}'] = y_pred
        input_df.to_csv(predict_result, index=False)
        print(f"Predictions appended to: {predict_result} (column: {rule})")

        # Save model
        rule_safe = rule.replace('/', '_').replace('\\', '_')
        model_path = os.path.join(f'{OUTPUT}/model/{rule_safe}.pkl')
        joblib.dump(clf_xgb, model_path)
        print(f"Model saved: {model_path}")

    print("All scenarios processed")