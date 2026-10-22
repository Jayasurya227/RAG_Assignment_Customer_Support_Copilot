"""
Evaluation Step 5: Comparison Table Generator
Section 8.4 Implementation

Aggregates all evaluation results across RAG pipeline versions:
- V1 (Basic RAG Baseline)
- V2 Single-Index (Advanced RAG: Hybrid + Reranking)
- V2 Multi-Index (Domain Partitioning + Intent Routing)

Outputs the official Section 8.4 summary comparison table to:
- evaluation/comparison_table.csv
- Formatted console display
"""

import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Suppress framework warnings
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
CSV_OUT_PATH = EVAL_DIR / "comparison_table.csv"

# Import calculation functions from step 2 and step 3
import importlib.util

def load_metrics_modules():
    retrieval_path = EVAL_DIR / "2_retrieval_metrics.py"
    spec_r = importlib.util.spec_from_file_location("retrieval_metrics_mod", retrieval_path)
    r_mod = importlib.util.module_from_spec(spec_r)
    spec_r.loader.exec_module(r_mod)

    gen_path = EVAL_DIR / "3_generation_metrics.py"
    spec_g = importlib.util.spec_from_file_location("gen_metrics_mod", gen_path)
    g_mod = importlib.util.module_from_spec(spec_g)
    spec_g.loader.exec_module(g_mod)

    return r_mod.calculate_metrics, g_mod.evaluate_generation_metrics


def build_comparison_table() -> list[dict[str, Any]]:
    """Builds the comprehensive comparison table comparing all 3 versions."""
    calculate_retrieval, calculate_gen = load_metrics_modules()

    versions = [
        ("V1 (Basic RAG)", RESULTS_DIR / "v1_results.json"),
        ("V2 Single-Index (Advanced RAG)", RESULTS_DIR / "v2_single_index_results.json"),
        ("V2 Multi-Index (Partitioned + Router)", RESULTS_DIR / "v2_multi_index_results.json"),
    ]

    version_data = {}
    for ver_name, path in versions:
        if path.exists():
            r_metrics = calculate_retrieval(path)
            g_metrics = calculate_gen(path)
            # Combine
            combined = {**r_metrics, **g_metrics}
            version_data[ver_name] = combined
        else:
            version_data[ver_name] = None

    # Define rows matching Section 8.4 requirements
    rows = [
        ("Precision@5", "precision_at_5", "{:.4f}"),
        ("Recall@5", "recall_at_5", "{:.4f}"),
        ("Multi-Source Coverage", "coverage_multi_source", "{:.4f}"),
        ("Refusal Accuracy (%)", "refusal_accuracy_pct", "{:.1f}%"),
        ("False Refusals (count)", "false_refusals", "{}"),
        ("Routing Accuracy (%)", "routing_accuracy_pct", lambda v, name: "n/a (no router)" if "v1" in name.lower() or "single" in name.lower() else f"{v:.1f}%"),
        ("Faithfulness", "faithfulness", "{:.4f}"),
        ("Answer Relevance", "answer_relevance", "{:.4f}"),
    ]

    table_data = []
    headers = ["Metric", "V1 (Basic RAG)", "V2 Single-Index", "V2 Multi-Index"]

    print("=" * 85)
    print("DANDES ELECTRONICS COPILOT - MASTER EVALUATION COMPARISON TABLE (Section 8.4)")
    print("=" * 85)
    header_line = f"{headers[0]:<28} | {headers[1]:<16} | {headers[2]:<16} | {headers[3]:<16}"
    print(header_line)
    print("-" * 85)

    csv_rows = [headers]

    for label, key, fmt in rows:
        v1_val = version_data.get("V1 (Basic RAG)")
        v2s_val = version_data.get("V2 Single-Index (Advanced RAG)")
        v2m_val = version_data.get("V2 Multi-Index (Partitioned + Router)")

        def format_val(val_dict, v_name):
            if val_dict is None:
                return "Pending Run"
            raw = val_dict.get(key)
            if raw is None:
                return "N/A"
            if callable(fmt):
                return fmt(raw, v_name)
            return fmt.format(raw)

        v1_str = format_val(v1_val, "V1 (Basic RAG)")
        v2s_str = format_val(v2s_val, "V2 Single-Index")
        v2m_str = format_val(v2m_val, "V2 Multi-Index")

        line = f"{label:<28} | {v1_str:<16} | {v2s_str:<16} | {v2m_str:<16}"
        print(line)
        csv_rows.append([label, v1_str, v2s_str, v2m_str])

    print("=" * 85)

    # Save to CSV
    with open(CSV_OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\nSaved official comparison table to: {CSV_OUT_PATH.relative_to(BASE_DIR)}")
    return csv_rows


if __name__ == "__main__":
    build_comparison_table()
