"""
Evaluation harness — run with: python eval/run_eval.py
Or via Makefile: make eval
Prints a scorecard comparing the delta engine output against labeled ground truth.
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import compute_delta_metrics, compute_chat_metrics
from src.delta_engine.engine import run_delta
from src.ingestion.canonicalizer import canonicalize
from src.reporting.generator import generate_report
from src.chat.answer import get_grounded_answer
from src.observability.logging import get_logger, set_correlation_id

logger = get_logger(__name__)

DATASET_DIR = Path(__file__).parent / "datasets"
EVAL_JOB_ID_PREFIX = "eval"


def run_eval():
    dataset_files = list(DATASET_DIR.glob("*.json"))
    if not dataset_files:
        print("❌ No dataset files found in eval/datasets/")
        sys.exit(1)

    all_delta_metrics = []
    all_chat_metrics = []
    failure_cases = []

    for dataset_file in dataset_files:
        pairs = json.loads(dataset_file.read_text(encoding="utf-8"))

        for pair in pairs:
            pair_id = pair["pair_id"]
            job_id = f"{EVAL_JOB_ID_PREFIX}-{pair_id}"
            set_correlation_id(job_id)

            print(f"\n{'='*60}")
            print(f"Evaluating pair: {pair_id}")
            print(f"  Rev A: {pair['pid_a_path']}")
            print(f"  Rev B: {pair['pid_b_path']}")

            # Check files exist — if not, log as failure and continue
            a_exists = Path(pair["pid_a_path"]).exists()
            b_exists = Path(pair["pid_b_path"]).exists()
            if not a_exists or not b_exists:
                failure_cases.append({
                    "pair_id": pair_id,
                    "error": f"Sample files not found. "
                             f"A exists: {a_exists}, B exists: {b_exists}",
                })
                print(f"  ⚠️  SKIPPED — sample files not found. "
                      f"Add PDFs to data/samples/ with provenance notes.")
                continue

            try:
                # Run the pipeline
                entities_a, _ = canonicalize(pair["pid_a_path"])
                entities_b, _ = canonicalize(pair["pid_b_path"])
                delta_entries = run_delta(entities_a, entities_b)
                report_bundle = generate_report(delta_entries, job_id)

                # Parse predicted deltas from the JSON report
                report_json = json.loads(report_bundle.json_report)
                predicted = report_json.get("changes", [])

                # Delta metrics
                gt_deltas = pair.get("ground_truth_deltas", [])
                delta_metrics = compute_delta_metrics(predicted, gt_deltas)
                all_delta_metrics.append(delta_metrics)

                print(f"  Delta P/R/F1: "
                      f"{delta_metrics['precision']:.2f} / "
                      f"{delta_metrics['recall']:.2f} / "
                      f"{delta_metrics['f1']:.2f}")
                print(f"  TP={delta_metrics['true_positives']} "
                      f"FP={delta_metrics['false_positives']} "
                      f"FN={delta_metrics['false_negatives']}")

                # Chat metrics (requires Pinecone to be seeded — skip if not)
                gt_qa = pair.get("ground_truth_qa", [])
                if gt_qa:
                    try:
                        from src.chat.index import index_chunks
                        index_chunks(report_bundle.rag_chunks, job_id, "delta_report")

                        answers = []
                        keywords_list = []
                        for qa in gt_qa:
                            response = get_grounded_answer(qa["question"], job_id)
                            answers.append(response.answer)
                            keywords_list.append(qa["expected_keywords"])

                        chat_metrics = compute_chat_metrics(answers, keywords_list)
                        all_chat_metrics.append(chat_metrics)
                        print(f"  Chat keyword coverage: "
                              f"{chat_metrics['avg_keyword_coverage']:.2%}")
                    except Exception as chat_err:
                        print(f"  ⚠️  Chat eval skipped: {chat_err}")
                        failure_cases.append({
                            "pair_id": pair_id,
                            "stage": "chat_eval",
                            "error": str(chat_err),
                        })

            except Exception as e:
                failure_cases.append({
                    "pair_id": pair_id,
                    "stage": "pipeline",
                    "error": str(e),
                })
                print(f"  ❌ FAILED: {e}")

    # ── Final Scorecard ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SCORECARD")
    print(f"{'='*60}")

    if all_delta_metrics:
        avg_p = sum(m["precision"] for m in all_delta_metrics) / len(all_delta_metrics)
        avg_r = sum(m["recall"] for m in all_delta_metrics) / len(all_delta_metrics)
        avg_f1 = sum(m["f1"] for m in all_delta_metrics) / len(all_delta_metrics)
        print(f"Delta Engine  — Avg Precision: {avg_p:.2f} | "
              f"Recall: {avg_r:.2f} | F1: {avg_f1:.2f}")
    else:
        print("Delta Engine  — No pairs evaluated.")

    if all_chat_metrics:
        avg_cov = sum(m["avg_keyword_coverage"] for m in all_chat_metrics) / len(all_chat_metrics)
        print(f"Grounded Chat — Avg Keyword Coverage: {avg_cov:.2%}")
    else:
        print("Grounded Chat — Not evaluated (no Pinecone or no QA pairs).")

    if failure_cases:
        print(f"\nFailure cases ({len(failure_cases)}):")
        for f in failure_cases:
            print(f"  [{f['pair_id']}] {f.get('stage', 'unknown')}: {f['error']}")
    else:
        print("\n✅ No failures.")


if __name__ == "__main__":
    run_eval()
