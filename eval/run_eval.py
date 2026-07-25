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
from src.core.models import (
    EvalReportResponse,
    PairEvalResult,
    DeltaMetrics,
    ChatMetrics,
    FailureCase,
)

logger = get_logger(__name__)

DATASET_DIR = Path(__file__).parent / "datasets"
EVAL_JOB_ID_PREFIX = "eval"


def generate_eval_report() -> EvalReportResponse:
    dataset_files = list(DATASET_DIR.glob("*.json"))
    if not dataset_files:
        raise ValueError("No dataset files found in eval/datasets/")

    all_delta_metrics = []
    all_chat_metrics = []
    failure_cases = []
    pair_results = []

    for dataset_file in dataset_files:
        pairs = json.loads(dataset_file.read_text(encoding="utf-8"))

        for pair in pairs:
            pair_id = pair["pair_id"]
            job_id = f"{EVAL_JOB_ID_PREFIX}-{pair_id}"
            set_correlation_id(job_id)

            a_exists = Path(pair["pid_a_path"]).exists()
            b_exists = Path(pair["pid_b_path"]).exists()
            if not a_exists or not b_exists:
                failure_cases.append(FailureCase(
                    pair_id=pair_id,
                    stage="file_check",
                    error=f"Sample files not found. A exists: {a_exists}, B exists: {b_exists}",
                ))
                pair_results.append(PairEvalResult(
                    pair_id=pair_id,
                    status="skipped (files not found)"
                ))
                continue

            delta_metrics_obj = None
            chat_metrics_obj = None
            status = "success"

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
                delta_metrics_raw = compute_delta_metrics(predicted, gt_deltas)
                delta_metrics_obj = DeltaMetrics(**delta_metrics_raw)
                all_delta_metrics.append(delta_metrics_obj)

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

                        chat_metrics_raw = compute_chat_metrics(answers, keywords_list)
                        chat_metrics_obj = ChatMetrics(**chat_metrics_raw)
                        all_chat_metrics.append(chat_metrics_obj)
                    except Exception as chat_err:
                        failure_cases.append(FailureCase(
                            pair_id=pair_id,
                            stage="chat_eval",
                            error=str(chat_err),
                        ))
                        status = "partial_success (chat failed)"

            except Exception as e:
                failure_cases.append(FailureCase(
                    pair_id=pair_id,
                    stage="pipeline",
                    error=str(e),
                ))
                status = "failed"

            pair_results.append(PairEvalResult(
                pair_id=pair_id,
                delta_metrics=delta_metrics_obj,
                chat_metrics=chat_metrics_obj,
                status=status
            ))

    avg_p = sum(m.precision for m in all_delta_metrics) / len(all_delta_metrics) if all_delta_metrics else 0.0
    avg_r = sum(m.recall for m in all_delta_metrics) / len(all_delta_metrics) if all_delta_metrics else 0.0
    avg_f1 = sum(m.f1 for m in all_delta_metrics) / len(all_delta_metrics) if all_delta_metrics else 0.0
    avg_cov = sum(m.avg_keyword_coverage for m in all_chat_metrics) / len(all_chat_metrics) if all_chat_metrics else 0.0

    return EvalReportResponse(
        overall_delta_precision=avg_p,
        overall_delta_recall=avg_r,
        overall_delta_f1=avg_f1,
        overall_chat_keyword_coverage=avg_cov,
        pair_results=pair_results,
        failures=failure_cases,
    )


def run_eval():
    try:
        report = generate_eval_report()
    except Exception as e:
        print(f"❌ Eval failed to run: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("SCORECARD")
    print(f"{'='*60}")

    if report.pair_results:
        print(f"Delta Engine  — Avg Precision: {report.overall_delta_precision:.2f} | "
              f"Recall: {report.overall_delta_recall:.2f} | F1: {report.overall_delta_f1:.2f}")
    else:
        print("Delta Engine  — No pairs evaluated.")

    if report.overall_chat_keyword_coverage > 0:
        print(f"Grounded Chat — Avg Keyword Coverage: {report.overall_chat_keyword_coverage:.2%}")
    else:
        print("Grounded Chat — Not evaluated (no Pinecone or no QA pairs).")

    if report.failures:
        print(f"\nFailure cases ({len(report.failures)}):")
        for f in report.failures:
            print(f"  [{f.pair_id}] {f.stage}: {f.error}")
    else:
        print("\n✅ No failures.")


if __name__ == "__main__":
    run_eval()
