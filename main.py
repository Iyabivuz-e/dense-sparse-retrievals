"""
Paper replicate: "Operational Advice for Dense and Sparse Retrievers" (Jimmy Lin)

Usage:
    python run_experiments.py                                      # Run ALL
    python run_experiments.py --datasets nfcorpus                  # One dataset
    python run_experiments.py --datasets nfcorpus fiqa             # Two datasets
    python run_experiments.py --methods bm25 splade                # Specific methods
    python run_experiments.py --datasets nfcorpus --methods bm25   # Smoke test
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import config
from data.loader import load_dataset
from evaluation.metrics import compute_ndcg
from retrievers.bm25 import BM25Retriever
from retrievers.splade import SPLADERetriever
from retrievers.flat import FlatRetriever
from retrievers.hnsw import HNSWRetriever


# Each lambda creates a FRESH retriever instance per trial.
RETRIEVERS = {
    "bm25":           lambda: BM25Retriever(),
    "splade":         lambda: SPLADERetriever(),
    "flat":           lambda: FlatRetriever(quantize=False),
    "flat_quantized": lambda: FlatRetriever(quantize=True),
    "hnsw":           lambda: HNSWRetriever(quantize=False),
    "hnsw_quantized": lambda: HNSWRetriever(quantize=True),
}


def get_num_trials(method_name: str) -> int:
    """HNSW is non-deterministic - we used multiple trials.  Everything else goes to 1."""
    if "hnsw" in method_name:
        return config.HNSW_TRIALS
    return 1


def run_single_experiment(
    retriever, corpus, queries, qrels, dataset_name, method_name, trial
) -> dict:
    """Run one (dataset, method, trial) combination and return the record."""

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name} | Method: {method_name} | Trial: {trial}")
    print(f"{'='*60}")

    # Build index 
    print("  Building index...")
    index_time = retriever.build_index(corpus, dataset_name)
    print(f"  Index built in {index_time:.2f}s")

    # Search 
    print(f" Searching ({len(queries)} queries, top_k={config.TOP_K})...")
    results, qps = retriever.search(queries, top_k=config.TOP_K)
    print(f"  QPS: {qps:.2f}")

    # Evaluate: ndcg@k
    print(" Evaluating...")
    ndcg = compute_ndcg(qrels, results)
    print(f"  NDCG@10: {ndcg['NDCG@10']:.5f}")

    record = {
        "dataset":      dataset_name,
        "method":       method_name,
        "trial":        trial,
        "num_docs":     len(corpus),
        "num_queries":  len(queries),
        "index_time_s": round(index_time, 4),
        "qps":          round(qps, 4),
        **ndcg,         # NDCG@1, NDCG@10, NDCG@100
    }
    return record


def save_results(all_results: list, results_dir: str):
    """Save results to both CSV and JSON."""
    os.makedirs(results_dir, exist_ok=True)
    raw_dir = os.path.join(results_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # CSV summary 
    csv_path = os.path.join(results_dir, "summary.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    print(f"\n Summary saved → {csv_path}")

    # Raw JSON - timestamped for history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(raw_dir, f"run_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f" Raw results saved → {json_path}")


def print_summary_table(all_results: list):
    """Print a formatted summary table to the terminal."""
    header = (
        f"{'Dataset':<12} {'Method':<18} {'Trial':>5} "
        f"{'Index(s)':>10} {'QPS':>12} {'NDCG@10':>10}"
    )
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")
    for r in all_results:
        print(
            f"{r['dataset']:<12} {r['method']:<18} {r['trial']:>5} "
            f"{r['index_time_s']:>10.2f} {r['qps']:>12.2f} "
            f"{r['NDCG@10']:>10.5f}"
        )
    print(f"{'='*len(header)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run dense vs. sparse retrieval experiments"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=config.DATASETS,
        help=f"Datasets to evaluate (default: {config.DATASETS})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=config.METHODS,
        help=f"Methods to evaluate (default: {config.METHODS})",
    )
    args = parser.parse_args()

    # Validate methods
    for m in args.methods:
        if m not in RETRIEVERS:
            raise ValueError(
                f"Unknown method: '{m}'. Choose from: {list(RETRIEVERS.keys())}"
            )

    all_results = []
    total_start = time.time()

    print(f" Starting experiments")
    print(f"   Datasets : {args.datasets}")
    print(f"   Methods  : {args.methods}")
    print(f"   Top-K    : {config.TOP_K}")
    print(f"   HNSW trials : {config.HNSW_TRIALS}")

    for dataset_name in args.datasets:
        # Load dataset (download if needed) 
        print(f"\n Loading dataset: {dataset_name}")
        corpus, queries, qrels = load_dataset(dataset_name, config.DATASETS_DIR)
        print(f"   {len(corpus)} docs, {len(queries)} queries")

        for method_name in args.methods:
            n_trials = get_num_trials(method_name)

            for trial in range(n_trials):
                # Fresh retriever per trial (important for HNSW non-determinism)
                retriever = RETRIEVERS[method_name]()

                record = run_single_experiment(
                    retriever, corpus, queries, qrels,
                    dataset_name, method_name, trial,
                )
                all_results.append(record)


    # total_time = time.time() - total_start
    # print(f"\n⏱️  Total experiment time: {total_time:.1f}s")

    save_results(all_results, config.RESULTS_DIR)
    print_summary_table(all_results)


if __name__ == "__main__":
    main()