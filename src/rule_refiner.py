import argparse
import asyncio
import copy
import json
import os
import random
import re
import aiohttp
from typing import List, Dict, Any, Callable
from tqdm import tqdm
import shutil

from experience import ExperienceUpdater
from setting.prompts import *
from setting.config import *

def load_rollouts(rollout_filename: str) -> list[dict]:
    """
    Loads rollout results from a JSONL file.
    """
    results = []
    if os.path.exists(rollout_filename):
        with open(rollout_filename, encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
    return results


def extract_tag_content(text: str, tag: str) -> str | None:
    """从文本中提取指定XML标签内的内容"""
    if not isinstance(text, str):
        return None
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def new_verify_func(prediction: str, ground_truth: str) -> bool:
    """
    新的验证函数：提取 <result> 标签内容与 ground_truth 比较
    """
    extracted_result = extract_tag_content(prediction, "result")
    if extracted_result is None:
        return False
    return extracted_result == ground_truth

async def local_rollout_task(session: aiohttp.ClientSession, task_data: Dict[str, Any], temp: float, max_tokens: int) -> str:
    """单个异步模型调用任务"""
    payload = {
        "model": LLM_PATH,
        "messages": [{"role": "user", "content": [{"type": "text", "text": task_data["prompt"]}]}],
        **LLM_PARAMS
    }
    try:
        async with session.post(LLM_URL, headers=HEADERS, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"API call failed for task {task_data.get('problem', 'N/A')}: {e}")
        return ""

async def local_rollout_dataset(
    data: List[Dict[str, Any]],
    verify_func: Callable[[str, str], bool],
    rollout_concurrency: int,
    temperature: float,
    max_tokens: int
) -> (List[Dict[str, Any]], Dict[str, Any]):
    """
    使用本地部署模型进行并发 rollout 的新函数
    """
    tasks = []
    connector = aiohttp.TCPConnector(limit=rollout_concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        for item in data:
            tasks.append(local_rollout_task(session, item, temperature, max_tokens))
        
        predictions = []
        # 使用tqdm来显示异步任务的进度
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Student model rollout"):
            predictions.append(await f)

    rollouts = []
    correct_count = 0
    # 注意：predictions的顺序可能与data不完全对应，这里假设as_completed保持了某种程度的顺序
    # 为了100%严谨，应该将任务和结果关联起来，但对于当前实现，我们暂时这样处理
    if len(predictions) != len(data):
        print(f"Warning: Number of predictions ({len(predictions)}) does not match data size ({len(data)}). This might cause issues.")
        # Handle mismatch if necessary, e.g., by returning early
        return [], {}

    for i, item in enumerate(data):
        prediction = predictions[i]
        is_correct = verify_func(prediction, item["groundtruth"])
        if is_correct:
            correct_count += 1
        
        rollout_item = copy.deepcopy(item)
        rollout_item['prediction'] = prediction
        rollout_item['trajectories'] = [{
            "trajectory": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": prediction}
            ]
        }]
        rollout_item['reward'] = 1 if is_correct else 0
        rollout_item['correct'] = is_correct
        rollouts.append(rollout_item)

    stats = {
        "total": len(data),
        "correct": correct_count,
        "accuracy": correct_count / len(data) if len(data) > 0 else 0
    }
    print(f"Rollout stats: {stats}")
    return rollouts, stats


async def main(args):

    dataset_name = args.dataset_name
    experiment_name = args.experiment_name

    experiment_dir = f"./result/{experiment_name}/rule_refiner"
    dataset_dir = f"./data/{dataset_name}/train.json"
    os.makedirs(experiment_dir, exist_ok=True)

    primary_src = (f"./result/{experiment_name}/kdaro_rule.json")

    if os.path.exists(primary_src):
        src_file = primary_src
    else:
        src_file = f"./data/{dataset_name}/rule.json"
    target_dir = f"{experiment_dir}/step_0"
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "experiences.json")
    shutil.copy2(src_file, target_file)

    with open(dataset_dir, 'r', encoding='utf-8-sig') as f:
        original_data = json.load(f)

    with open(f"./data/{dataset_name}/info.json", "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        INFO_COLUMN = data["INFO_COLUMN"]
        QUESTION_COLUMN = data["QUESTION_COLUMN"]
        RULE_COLUMN = data["RULE_COLUMN"]
        ANSWER_COLUMN = data["ANSWER_COLUMN"]

    with open(f"./data/{dataset_name}/rule.json", "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # 转换键名以匹配所需的格式
    train_data = [
        {"problem": REFINE_PROMPT_1 + str(data) + REFINE_PROMPT_2 + item[INFO_COLUMN] + item[QUESTION_COLUMN], "groundtruth": item[ANSWER_COLUMN],
         "rule": item[RULE_COLUMN]}
        for item in original_data
    ]

    print(f"Loaded {len(train_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- truncated to {args.dataset_truncate}")
        train_data = train_data[: args.dataset_truncate]
    
    if len(train_data) > 0 and len(train_data) % args.batchsize != 0:
        print(f"Warning: Dataset size ({len(train_data)}) is not divisible by batchsize ({args.batchsize}). Truncating to the nearest multiple.")
        new_size = len(train_data) - (len(train_data) % args.batchsize)
        train_data = train_data[:new_size]
        print(f"- new truncated size: {new_size}")

    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        stats = json.load(open(stats_filename))
    else:
        stats = {}

    for epoch in range(args.epochs):
        print("=" * 30 + f"\nEpoch {epoch}\n" + "=" * 30)
        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)

        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_data = []
            with open(shuffled_filename, encoding='utf-8') as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
            print(f"Loaded {len(shuffled_data)} records from shuffled data")
        else:
            print(f"Shuffling data ...")
            shuffled_data = copy.deepcopy(train_data)
            random.shuffle(shuffled_data)
            with open(shuffled_filename, "w", encoding='utf-8') as f:
                for each in shuffled_data:
                    f.write(json.dumps(each, ensure_ascii=False) + "\n")

        num_batches = len(shuffled_data) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if f"step_{step}" in stats and stats.get(f"step_{step}", {}).get("complete", False):
                print(f"Step {step} already complete, skipping.")
                continue

            print(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)
            
            batch_data = copy.deepcopy(shuffled_data[batch_idx * args.batchsize : (batch_idx + 1) * args.batchsize])

            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            if os.path.exists(rollout_filename):
                 print(f"Rollout file for step {step} exists, skipping to experience generation.")
                 rollouts = load_rollouts(rollout_filename)
            else:
                if step > 0:
                    experience_filename = os.path.join(experiment_dir, f"step_{step}", "experiences.json")
                    if os.path.exists(experience_filename):
                        with open(experience_filename, 'r', encoding='utf-8') as f:
                            experiences = json.load(f)
                    else:
                        experiences = {}
                else:
                    experiences = {}
                
                formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ])
                formatted_batch_data = [{
                    "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                        experiences=formatted_experiences if formatted_experiences else "None",
                        problem=each["problem"],
                    ),
                    **each
                } for each in batch_data]
                
                print(f"GRPO rollout number={args.grpo_n}")
                formatted_batch_data = formatted_batch_data * args.grpo_n

                rollouts, rollout_stats = await local_rollout_dataset(
                    data=formatted_batch_data,
                    verify_func=new_verify_func,
                    rollout_concurrency=args.rollout_concurrency,
                    temperature=args.rollout_temperature,
                    max_tokens=args.rollout_max_tokens,
                )
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False, "rollout": rollout_stats}
                
                with open(rollout_filename, 'w', encoding='utf-8') as f:
                    for r in rollouts:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')

            next_step_dir = os.path.join(experiment_dir, f"step_{step+1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(next_step_dir, "experiences.json")

            if os.path.exists(next_experience_filename):
                print(f"Experiences for next step ({step+1}) already exist, skipping experience update.")
            else:
                if step > 0:
                    experience_filename = os.path.join(experiment_dir, f"step_{step}", "experiences.json")
                    if os.path.exists(experience_filename):
                        with open(experience_filename, 'r', encoding='utf-8') as f:
                            experiences = json.load(f)
                    else:
                        experiences = {}
                else:
                    experiences = {}

                new_experiences = await ExperienceUpdater().run(
                    rollouts=rollouts, 
                    experiences=experiences,
                    save_dir=cur_step_dir,
                    max_workers=args.rollout_concurrency,
                    given_ground_truth=True if args.given_ground_truth=="True" else False,
                    only_partial_correct=False,
                )
                
                with open(next_experience_filename, "w", encoding='utf-8') as f:
                    json.dump(new_experiences, f, indent=2, ensure_ascii=False)
                
                print(f"Saved {len(new_experiences)} total experiences to {next_experience_filename}")

            stats[f"step_{step}"]["complete"] = True
            with open(stats_filename, "w", encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule Refiner")
    parser.add_argument("--experiment_name", type=str, required=True, help="name of experiment run")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset")
    parser.add_argument("--dataset_truncate", type=int, default=None, help="Truncate dataset to first N samples")
    parser.add_argument("--given_ground_truth", type=str, default=True, help="Whether use ground truth answers")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=4, help="batchsize")
    parser.add_argument("--grpo_n", type=int, default=5, help="number of rollouts in a group of GRPO")
    parser.add_argument("--rollout_concurrency", type=int, default=20, help="Concurrency level for rollouts")
    parser.add_argument("--rollout_temperature", type=float, default=0.7, help="Temperature for the LLM")
    parser.add_argument("--rollout_max_tokens", type=int, default=5172, help="Max tokens for each rollout batch")

    args = parser.parse_args()
    asyncio.run(main(args))