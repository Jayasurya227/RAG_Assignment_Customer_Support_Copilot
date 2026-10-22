"""
Evaluation Step 3: Retrieval Metrics
Calculates retrieval metrics on pipeline evaluation results as specified in Section 8.3:
- Precision@5: (Relevant docs in top 5) / 5
- Recall@5: (Relevant docs in top 5) / (Total expected sources)
- Coverage: Multi-source questions only: (Expected sources found) / (Expected sources needed)
- Routing Accuracy: % of questions where router searched all expected indexes and at most one extra index
- Refusal Accuracy: % of out-of-scope questions answered with exact fixed refusal sentence
- False Refusals: Count of in-scope questions wrongly answered with refusal
"""

import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
EXACT_REFUSAL = "I don't know based on the available information."


def calculate_metrics(results_path: Path) -> dict[str, Any]:
    """Calculate all Section 8.3 retrieval metrics for a results JSON file."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    in_scope_queries = []
    out_of_scope_queries = []

    for item in results:
        expected_sources = item.get("expected_sources", [])
        if expected_sources:
            in_scope_queries.append(item)
        else:
            out_of_scope_queries.append(item)

    # 1. Precision@5, Recall@5, Coverage on in-scope questions
    precisions = []
    recalls = []
    multi_source_coverages = []
    false_refusals = 0

    for item in in_scope_queries:
        expected = set(item.get("expected_sources", []))
        # Top 5 unique retrieved doc IDs
        retrieved = set(item.get("retrieved_doc_ids", [])[:5])
        generated_answer = item.get("generated_answer", "")

        # Check for false refusal
        if EXACT_REFUSAL.lower() in generated_answer.lower():
            false_refusals += 1

        # Relevant docs in top 5
        relevant_found = expected.intersection(retrieved)
        
        # Precision@5 = relevant / 5
        precisions.append(len(relevant_found) / 5.0)

        # Recall@5 = relevant / total expected
        if expected:
            recalls.append(len(relevant_found) / len(expected))

        # Coverage for multi-source questions (expected >= 2)
        if len(expected) >= 2:
            multi_source_coverages.append(len(relevant_found) / len(expected))

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_coverage = (
        sum(multi_source_coverages) / len(multi_source_coverages)
        if multi_source_coverages
        else 0.0
    )

    # 2. Refusal Accuracy on out-of-scope questions
    correct_refusals = 0
    for item in out_of_scope_queries:
        ans = item.get("generated_answer", "").strip()
        if ans == EXACT_REFUSAL or EXACT_REFUSAL.lower() in ans.lower():
            correct_refusals += 1

    refusal_accuracy = (
        (correct_refusals / len(out_of_scope_queries)) * 100.0
        if out_of_scope_queries
        else 100.0
    )

    # 3. Routing Accuracy (applicable when router is used)
    routing_correct = 0
    total_routing_tested = len(results)
    for item in results:
        expected_idx = set(item.get("expected_indexes", []))
        routing_info = item.get("routing_decision", {})
        predicted_idx = set(routing_info.get("indexes", []))
        intents = routing_info.get("intents", [])

        # For out-of-scope: correct only if out_of_scope intent or no index searched
        if not expected_idx:
            if "out_of_scope" in intents or not predicted_idx:
                routing_correct += 1
        else:
            # All expected indexes searched + at most one extra
            if expected_idx.issubset(predicted_idx) and len(predicted_idx - expected_idx) <= 1:
                routing_correct += 1

    routing_accuracy = (routing_correct / total_routing_tested) * 100.0 if total_routing_tested else 0.0

    return {
        "precision_at_5": round(avg_precision, 4),
        "recall_at_5": round(avg_recall, 4),
        "coverage_multi_source": round(avg_coverage, 4),
        "refusal_accuracy_pct": round(refusal_accuracy, 2),
        "false_refusals": false_refusals,
        "routing_accuracy_pct": round(routing_accuracy, 2),
        "total_queries": len(results),
        "in_scope_count": len(in_scope_queries),
        "out_of_scope_count": len(out_of_scope_queries),
    }


def print_metrics_summary(version_name: str, metrics: dict[str, Any]) -> None:
    """Pretty prints the metrics."""
    print("\n" + "=" * 60)
    print(f"Retrieval Metrics Summary: {version_name}")
    print("=" * 60)
    print(f"Total Questions Evaluated : {metrics['total_queries']} ({metrics['in_scope_count']} in-scope, {metrics['out_of_scope_count']} out-of-scope)")
    print(f"Precision@5               : {metrics['precision_at_5']:.4f}")
    print(f"Recall@5                  : {metrics['recall_at_5']:.4f}")
    print(f"Coverage (Multi-Source)   : {metrics['coverage_multi_source']:.4f}")
    print(f"Refusal Accuracy          : {metrics['refusal_accuracy_pct']:.1f}%")
    print(f"False Refusals            : {metrics['false_refusals']}")
    if "v1" in version_name.lower():
        print(f"Routing Accuracy          : n/a (single collection baseline)")
    else:
        print(f"Routing Accuracy          : {metrics['routing_accuracy_pct']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    target_file = RESULTS_DIR / "v1_results.json"
    if len(sys.argv) > 1:
        target_file = Path(sys.argv[1])

    if not target_file.exists():
        print(f"Results file does not exist yet: {target_file}")
        print("Please run evaluation/1_run_versions.py first.")
        sys.exit(1)

    m = calculate_metrics(target_file)
    print_metrics_summary(target_file.stem.upper(), m)
