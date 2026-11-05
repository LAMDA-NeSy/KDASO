import os
import sys
import subprocess
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.rule_generator import main as run_generator
from src.rule_retriever import main as run_retriever
from src.run import main as run_adjudication


def run_refiner_subprocess(
        experiment_name: str,
        dataset_name: str,
        epochs: int = 2,
        batchsize: int = 4,
        truncate: int = None,
        grpo_n: int = 5,
        concurrency: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 5172,
        given_ground_truth: str = "True"
):

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src", "rule_refiner.py"),
        "--experiment_name", experiment_name,
        "--dataset_name", dataset_name,
        "--epochs", str(epochs),
        "--batchsize", str(batchsize),
        "--grpo_n", str(grpo_n),
        "--rollout_concurrency", str(concurrency),
        "--rollout_temperature", str(temperature),
        "--rollout_max_tokens", str(max_tokens),
        "--given_ground_truth", given_ground_truth,
    ]

    if truncate is not None:
        cmd.extend(["--dataset_truncate", str(truncate)])

    print(f"\n{'=' * 60}")
    print(f"🚀 Rule Refiner (subprocess)")
    print(f"{'=' * 60}")
    print(f"命令: {' '.join(cmd)}\n")

    try:
        process = subprocess.run(cmd, check=True, text=True)
        print(f"\n✅ Rule Refiner Success！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Rule Refiner Failed: {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"\n❌ Error")
        return False


def main():
    parser = argparse.ArgumentParser(description="KDARO")

    # General Parameters
    parser.add_argument("--experiment_name", type=str, default="test_data" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="Experiment name")
    parser.add_argument("--dataset_name", type=str, default="test_data", help="Dataset name")

    # Generator Parameters
    parser.add_argument("--generator_top_k", type=int, default=4, help="Generator retrieval top-k")

    # Refiner Parameters
    parser.add_argument("--refiner_epochs", type=int, default=2, help="Refiner training epochs")
    parser.add_argument("--refiner_batchsize", type=int, default=4, help="Refiner batch size")
    parser.add_argument("--refiner_truncate", type=int, default=300, help="Refiner data truncation")
    parser.add_argument("--refiner_grpo_n", type=int, default=2, help="Refiner sampling count")
    parser.add_argument("--refiner_concurrency", type=int, default=128, help="Refiner concurrency")
    parser.add_argument("--refiner_temperature", type=float, default=0.7, help="Refiner temperature")
    parser.add_argument("--refiner_max_tokens", type=int, default=5172, help="Refiner max token count")
    parser.add_argument("--refiner_ground_truth", type=str, default="True", help="Whether to use ground truth")

    # Retriever Parameters
    parser.add_argument("--use_clustering", action="store_false", help="Use clustering")
    parser.add_argument("--true_sample_ratio", type=int, default=1, help="True sample ratio")

    # Execution Control
    parser.add_argument("--skip_generator", action="store_true", help="Skip Generator")
    parser.add_argument("--skip_refiner", action="store_true", help="Skip Refiner")
    parser.add_argument("--skip_retriever", action="store_true", help="Skip Retriever")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop execution on error")

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"🎯 Starting KDARO")
    print(f"{'=' * 60}")
    print(f"Experiment name：{args.experiment_name}")
    print(f"Dataset name：{args.dataset_name}")
    print(f"Starting time：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")
    os.makedirs(f"./result/{args.experiment_name}", exist_ok=True)

    execution_status = {
        "generator": "skipped" if args.skip_generator else "pending",
        "refiner": "skipped" if args.skip_refiner else "pending",
        "retriever": "skipped" if args.skip_retriever else "pending",
    }

    # ========== 1. Rule Generator ==========
    if not args.skip_generator:
        print(f"\n{'=' * 60}")
        print(f"Stage 1: Rule Generator")
        print(f"{'=' * 60}")

        try:
            run_generator(
                experiment_name = args.experiment_name,
                dataset_name = args.dataset_name,
                top_k=args.generator_top_k
            )
            execution_status["generator"] = "success"
            print(f"\n✅ Rule Generator Done！")
        except Exception as e:
            execution_status["generator"] = f"failed: {str(e)}"
            print(f"\n❌ Rule Generator Failed：{e}")
            if args.stop_on_error:
                print("\n⛔ Error")
                return False

    # ========== 2. Rule Refiner ==========
    if not args.skip_refiner:
        print(f"\n{'=' * 60}")
        print(f"Stage 2: Rule Refiner")
        print(f"{'=' * 60}")
        try:
            success = run_refiner_subprocess(
                experiment_name=args.experiment_name,
                dataset_name=args.dataset_name,
                epochs=args.refiner_epochs,
                batchsize=args.refiner_batchsize,
                truncate=args.refiner_truncate,
                grpo_n=args.refiner_grpo_n,
                concurrency=args.refiner_concurrency,
                temperature=args.refiner_temperature,
                max_tokens=args.refiner_max_tokens,
                given_ground_truth=args.refiner_ground_truth,
            )
            execution_status["refiner"] = "success" if success else f"failed"
            if not success and args.stop_on_error:
                print("\n⛔ Error")
                return False
            else:
                print(f"\n✅ Rule Refiner Done！")
        except Exception as e:
            execution_status["refiner"] = f"failed: {str(e)}"
            print(f"\n❌ Rule Refiner Failed：{e}")
            if args.stop_on_error:
                print("\n⛔ Error")
                return False

    # ========== 3. Rule Retriever ==========
    if not args.skip_retriever:
        print(f"\n{'=' * 60}")
        print(f"Stage 3: Rule Retriever")
        print(f"{'=' * 60}")
        try:
            run_retriever(experiment_name = args.experiment_name,
                dataset_name = args.dataset_name,
                use_clustering = args.use_clustering,
                true_sample_ratio = args.true_sample_ratio)
            execution_status["retriever"] = "success"
            print(f"\n✅ Rule Retriever Done！")
        except Exception as e:
            execution_status["retriever"] = f"failed: {str(e)}"
            print(f"\n❌ Rule Retriever Failed：{e}")
            if args.stop_on_error:
                print("\n⛔ Error")
                return False

    # ========== 4. Liability Adjudication ==========
    try:
        run_adjudication(experiment_name=args.experiment_name,
                      dataset_name=args.dataset_name)
        execution_status["adjudication"] = "success"
        print(f"\n✅ Liability Adjudication Done！")
    except Exception as e:
        execution_status["adjudication"] = f"failed: {str(e)}"
        print(f"\n❌ Liability Adjudication Failed：{e}")
        if args.stop_on_error:
            print("\n⛔ Error")
            return False


    # ========== Summary ==========
    print(f"\n{'=' * 60}")
    print(f"📊 Summary")
    print(f"{'=' * 60}")
    for module, status in execution_status.items():
        icon = "✅" if status == "success" else ("⏭️" if status == "skipped" else "❌")
        print(f"{icon} {module.capitalize()}: {status}")
    print(f"Ending time：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    return all(status == "success" or status == "skipped" for status in execution_status.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)