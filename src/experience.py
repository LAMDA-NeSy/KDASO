import json
import copy
import os
import asyncio
import aiohttp
import re

from collections import defaultdict
from tqdm import tqdm
from setting.config import *


def _extract_tag_content(text: str, tag: str) -> str | None:
    if not isinstance(text, str):
        return None
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class ExperienceUpdater:
    def __init__(self):
        pass

    async def _call_local_model_async(self, session, prompt: str, temperature: float = 0.0, max_tokens: int = 16384) -> str | None:
        """
        A dedicated asynchronous helper function to call the locally deployed model.
        """
        payload = {
            "model": LLM_PATH,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            **LLM_PARAMS
        }
        try:
            async with session.post(LLM_URL, headers=HEADERS, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Warning: API call failed. Error: {e}")
            return None

    async def run(self, rollouts, experiences, save_dir, max_workers=16, given_ground_truth=True, only_partial_correct=True):
        async with aiohttp.ClientSession() as session:
            problem_to_summarized_rollouts = await self._single_rollout_summary(
                session=session,
                rollouts=rollouts,
                save_dir=save_dir,
                max_workers=max_workers,
                given_ground_truth=given_ground_truth,
                only_partial_correct=only_partial_correct
            )

            critiques = await self._single_query_critique(
                session=session,
                problem_to_summarized_rollouts=problem_to_summarized_rollouts,
                experiences=experiences,
                save_dir=save_dir,
                max_workers=max_workers,
                given_ground_truth=given_ground_truth,
                only_partial_correct=only_partial_correct
            )

            new_experiences = await self._batch_update(
                session=session,
                experiences=experiences,
                critiques=critiques,
                save_dir=save_dir
            )

        new_experiences = {
            f"G{i}": exp for i, exp in enumerate(new_experiences.values())
        }
        return new_experiences

    async def _single_rollout_summary(
        self,
        session,
        rollouts,
        save_dir,
        max_workers,
        given_ground_truth=True,
        only_partial_correct=True
    ):
        filename = os.path.join(save_dir, "single_rollout_summary.json")
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, encoding='utf-8') as f:
                results = json.load(f)
                if len(results) > 0:
                    print("Single rollout summary")
                    print("- File exists, loaded from:", filename)
                    return results

        problems_to_rollouts = defaultdict(list)
        for each in rollouts:
            if "trajectories" in each and len(each["trajectories"]) > 0:
                problems_to_rollouts[each["problem"]].append(each)
        
        all_rollouts_to_process = []
        for rollouts_list in problems_to_rollouts.values():
            if given_ground_truth and only_partial_correct:
                scores = [each.get("reward", 0) for each in rollouts_list]
                avg_score = sum(scores) / len(scores) if scores else 0
                if 0 < avg_score < 1:
                    all_rollouts_to_process.extend(rollouts_list)
            else:
                all_rollouts_to_process.extend(rollouts_list)

        async def process_async(cur):
            grade_text = f"This solution trajectory yielded a **{'Correct' if cur.get('reward', 0) else 'Incorrect'}** answer."

            if given_ground_truth:
                model_judge = _extract_tag_content(cur.get('prediction', ''), 'rule') or 'Not extracted'
                model_answer = _extract_tag_content(cur.get('prediction', ''), 'result') or 'Not extracted'

                label_jf = cur.get('rule', 'None')

                prompt = f"""Please summarize the following solution trajectory.

Solution Trajectory:
{cur.get('trajectories', [{}])[0].get('trajectory', '')}

Evaluation:
{grade_text}

Ground Truth Liability Label:
{cur.get('groundtruth', '')}

Model-Generated Liability Label:
{model_answer}

**Model-Detected Scenario:**
{model_judge}

**Ground Truth Scenario:**
{label_jf}

Your summary should be concise and focus on key steps and potential errors. **Special Attention**: In addition to analyzing the reasons for any discrepancy between the model-generated liability label and the ground truth label, you should also analyze the differences between the 'Model-Detected Scenario' and the 'Ground Truth Scenario'."""

            else:
                prompt = f"""Please summarize the following solution trajectory.

Solution Trajectory:
{cur.get('trajectories', [{}])[0].get('trajectory', '')}"""

            response = await self._call_local_model_async(session, prompt)
            if response:
                return {"trajectory_summary": response, **cur}
            return None

        tasks = [process_async(cur) for cur in all_rollouts_to_process]
        
        results_list = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Single rollout summary"):
            results_list.append(await f)

        results = defaultdict(list)
        for result in results_list:
            if result:
                problem = result["problem"]
                results[problem].append(result)

        with open(filename, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    async def _single_query_critique(
        self,
        session,
        problem_to_summarized_rollouts,
        experiences,
        save_dir,
        max_workers,
        max_operations=1,
        given_ground_truth=True,
        only_partial_correct=True
    ):
        filename = os.path.join(save_dir, "single_query_critique.json")
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, encoding='utf-8') as f:
                results = json.load(f)
                if len(results) > 0:
                    print("Single query critique")
                    print("- File exists, loaded from:", filename)
                    return results

        all_rollouts_groups = []
        for rollouts in problem_to_summarized_rollouts.values():
            if given_ground_truth and only_partial_correct:
                scores = [each.get("reward", 0) for each in rollouts]
                avg_score = sum(scores) / len(scores) if scores else 0
                if 0 < avg_score < 1:
                    all_rollouts_groups.append(rollouts)
            else:
                all_rollouts_groups.append(rollouts)

        async def process_async(rollouts_per_problem):
            problem = rollouts_per_problem[0]["problem"]
            answer = rollouts_per_problem[0]["groundtruth"]
            label_jf = rollouts_per_problem[0].get("rule", "None")

            formatted_trajectories = "\n\n".join([
                (f"Trajectory {i + 1} (Answer: {'Correct' if each.get('reward', 0) else 'Incorrect'}):\n"
                 f"Model-Detected Scenario: {_extract_tag_content(each.get('prediction', ''), 'rule') or 'Not extracted'}\n"
                 f"Model-Generated Liability Answer: {_extract_tag_content(each.get('prediction', ''), 'result') or 'Not extracted'}\n"
                 f"Summary:\n{each.get('trajectory_summary', '')}")
                for i, each in enumerate(rollouts_per_problem)
            ])
            formatted_experiences = "\n".join([f"[{i}]. {e}" for i, e in experiences.items()]) if experiences else "无"
            
            json_example = """
[
  {
    "option": "add",
    "experience": "Specific content of the new experience to be added."
  },
  {
    "option": "modify",
    "modified_from": "R1",
    "experience": "Specific content of the modified experience for R1."
  }
]
"""

            prompt = f"""You are an analysis expert. Your task is to analyze solution trajectories and generate optimization experiences.

**Instructions:**
1. Analyze the "Solution Trajectories" provided below in detail.
2. **Special Attention**: Focus specifically on the discrepancies between the "Model-Detected Scenario" and the "Ground Truth Scenario".
3. Based on your analysis, propose up to {max_operations} operations to either add new rules or optimize existing ones. Each operation must be either "add" or "modify".
4. Your output **must** be a JSON list (array) that strictly adheres to the format example below. **Do not** create custom keys such as "analysis" or "optimization".

**Output Format Example:**
            ```json
            {json_example}
            ```

            **Information for Analysis:**

            Problem: {problem}

            Ground Truth Liability Answer: {answer}

            Ground Truth Scenario: {label_jf}

            Existing Experiences: {formatted_experiences}

            Solution Trajectories: {formatted_trajectories}

            **Final Instruction:** Please strictly follow the requirements above and provide your evaluation as a JSON list within a ```json ... ``` code block."""

            response = await self._call_local_model_async(session, prompt)
            if response:
                try:
                    operations = None
                    response_json_str = response.split("```json")[-1].split("```")[0]
                    parsed_json = json.loads(response_json_str)

                    if isinstance(parsed_json, list):
                        operations = parsed_json
                    elif isinstance(parsed_json, dict) and "optimization" in parsed_json:
                        print("Info: Model returned a custom dict format. Adapting...")
                        opt = parsed_json["optimization"]
                        operations = [{
                            "option": opt.get("action"),
                            "experience": opt.get("description")
                        }]
                    else:
                        raise TypeError("Model returned an unexpected JSON format.")
                    
                    return {"rollouts": rollouts_per_problem, "critique": response, "operations": operations[:max_operations]}
                
                except Exception as e:
                    print(f"Warning: failed to decode or process JSON in single query critique. Model response was: {response}. Error: {e}")
            return None

        tasks = [process_async(rollouts_group) for rollouts_group in all_rollouts_groups]
    
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Single query critique"):
            result = await f
            if result:
                results.append(result)

        with open(filename, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    async def _batch_update(
        self,
        session,
        experiences,
        critiques,
        save_dir,
        max_retries=3
    ):
        print("Batch update")
        filename = os.path.join(save_dir, "batch_update.json")
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, encoding='utf-8') as f:
                results = json.load(f)
            print("- File exists, loaded from:", filename)
            return results.get('new_experiences', {})

        all_operations = []
        for each in critiques:
            all_operations.extend(each.get("operations", []))
        
        if not all_operations:
            print("- No operations to process, skipping batch update.")
            with open(filename, "w", encoding='utf-8') as f:
                json.dump({"operations": [], "response": "Skipped", "revision_plan": [], "new_experiences": experiences}, f, indent=2, ensure_ascii=False)
            return experiences

        print("- Num of operations to process:", len(all_operations))
        
        candidate_experiences = copy.deepcopy(experiences)
        to_modify, max_ID = [], 0
        for op in all_operations:
            if op and op.get("option") == "modify" and op.get("modified_from") in candidate_experiences:
                to_modify.append(op)
            elif op and op.get("option") == "add":
                candidate_experiences[f"C{max_ID}"] = op.get("experience")
                max_ID += 1

        print("- Num of added experiences:", max_ID)
        print("- Num of experiences to be modified:", len(to_modify))
        print("- Num of candidate experiences:", len(candidate_experiences))

        response, revision_plan = None, []
        for _ in range(max_retries):
            # --- PROMPT ENHANCEMENT for _batch_update ---
            json_example = """
[
  {
    "option": "modify",
    "modified_from": "R1",
    "experience": "This is the updated content for experience R1."
  },
  {
    "option": "merge",
    "merged_from": ["R5", "R7"],
    "experience": "This is a new, more generalized experience created by merging experiences R5 and R7."
  }
] """
            prompt = f"""You are an Experience Repository Optimization Expert. Your task is to optimize the "Candidate Experiences" based on the "Requested Updates" and generate a "Revision Plan" in JSON format.

Instructions:

1. Carefully read the "Candidate Experiences" and the list of "Requested Updates".
2. For each "Requested Update", you may decide to either directly modify (modify) an existing experience or merge it with similar experiences to create a more generalized new experience (merge).
3. After carefully reviewing all "Candidate Experiences", you may also decide to merge several similar experiences into a single, more generalized new experience (merge).

Your output must be a JSON list (array) where each element represents an operation instruction. Do not return any natural language explanations outside of the JSON format.

Output Format Example:

```json
{json_example}
```

Information for Processing:

Candidate Experiences: {json.dumps(candidate_experiences, indent=2, ensure_ascii=False)}

Requested Updates: {json.dumps(to_modify, indent=2, ensure_ascii=False)}

Final Instruction: Please strictly follow the requirements above and provide your revision plan as a JSON list within a ```json ... ``` code block."""

            response = await self._call_local_model_async(session, prompt)
            if response:
                try:
                    revision_plan = json.loads(response.split("```json")[-1].split("```")[0])
                    break
                except Exception as e:
                    print(f"Warning: failed to decode in updating general experiences. Error: {e}")
        
        new_experiences = copy.deepcopy(candidate_experiences)
        for op in revision_plan:
            if not op: continue
            try:
                if op.get("option") == "modify":
                    new_experiences[op["modified_from"]] = op["experience"]
                elif op.get("option") == "merge":
                    for ID in op.get("merged_from", []):
                        if ID in new_experiences: del new_experiences[ID]
                    new_experiences[f"C{max_ID}"] = op.get("experience")
                    max_ID += 1
            except Exception as e:
                print(f"Error: failed to complete experience update: {op} | {e}")
        
        print("- Num of revised candidate experiences:", len(new_experiences))

        with open(filename, "w", encoding='utf-8') as f:
            json.dump({"operations": all_operations, "response": response, "revision_plan": revision_plan, "new_experiences": new_experiences}, f, indent=2, ensure_ascii=False)
        return new_experiences