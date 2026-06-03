````markdown
# Dense vs. Sparse Retrieval — Paper Replication

Replication of **"Operational Advice for Dense and Sparse Retrievers"** by Jimmy Lin et al.

This project compares sparse retrieval methods (BM25, SPLADE++) against dense retrieval methods (Flat index, HNSW) across multiple BEIR benchmark datasets, measuring retrieval quality (NDCG@k), throughput (QPS), and index build time.

## Project Structure

```
dense-sparse/
├── config.py                # Central configuration (paths, datasets, hyperparams)
├── main.py                  # Experiment runner — single entry point
│
├── data/
│   └── loader.py            # BEIR dataset downloader & loader
│
├── retrievers/
│   ├── base.py              # Abstract base class (common interface)
│   ├── bm25.py              # BM25 via Pyserini/Lucene
│   ├── splade.py            # SPLADE++ via Pyserini prebuilt indexes
│   ├── flat.py              # FAISS Flat (exact search)
│   └── hnsw.py              # FAISS HNSW (approximate search)
│
├── models/
│   └── embeddings.py        # Shared embedding computation & caching
│
├── evaluation/
│   └── metrics.py           # NDCG, MAP, Recall via BEIR but only returned NDCG
│
├── helpers/
│   └── converter.py         # Corpus format conversion utilities
│
├── datasets/                # Auto-downloaded BEIR datasets
├── indexes/                 # Built indexes (BM25 per dataset)
├── embeddings_cache/        # Cached corpus embeddings (dense methods)
├── results/                 # Experiment outputs (CSV + JSON)
│   ├── summary.csv
│   └── raw/
└── notebooks/               # Analysis & visualization
```

## Setup

### Prerequisites

- Python 3.11 (via Conda)
- Java 21 (`brew install openjdk@21` on macOS)

### Install Dependencies

```bash
conda activate dense-sparse
pip install -r requirements.txt
```

### Requirements

```
beir>=2.2.0
faiss-cpu>=1.13.2
pyserini>=1.6.0
sentence-transformers
```

## Usage

<<Quick Use>>
    python main.py                                      # Run ALL
    python main.py --datasets nfcorpus                  # One dataset
    python main.py --datasets nfcorpus fiqa             # Two datasets
    python main.py --methods bm25 splade                # Specific methods
    python main.py --datasets nfcorpus --methods bm25   # Smoke test


### Run all experiments

```bash
python main.py
```

### Run a single dataset

```bash
python main.py --datasets nfcorpus
```

### Run specific methods on specific datasets

```bash
python main.py --datasets nfcorpus fiqa --methods bm25 splade
```

### Quick smoke test

```bash
python main.py --datasets nfcorpus --methods bm25
```

## Methods

| Method | Type | Description |
|--------|------|-------------|
| `bm25` | Sparse | Pyserini Lucene inverted index |
| `splade` | Sparse | SPLADE++ CoCondenser-EnsembleDistil (prebuilt index) |
| `flat` | Dense | FAISS exact inner product search (IndexFlatIP) |
| `flat_quantized` | Dense | FAISS with Product Quantization (IndexPQ) |
| `hnsw` | Dense | FAISS approximate search (IndexHNSWFlat) |
| `hnsw_quantized` | Dense | FAISS HNSW + PQ (IndexHNSWPQ) |

## Datasets

Evaluated on [BEIR](https://github.com/beir-cellar/beir) benchmark datasets:

- **NFCorpus** — biomedical IR (3,633 docs)
- **FiQA** — financial QA (57,638 docs)
- **Quora** — duplicate question retrieval (522K docs)
- **NQ** — Natural Questions (2.6M docs)

Datasets are auto-downloaded on first run.

## Configuration

All settings are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | 1000 | Documents retrieved per query |
| `HNSW_TRIALS` | 5 | Number of trials for non-deterministic HNSW |
| `HNSW_M` | 16 | HNSW graph connectivity |
| `HNSW_EF_SEARCH` | 100 | HNSW search depth |
| `DENSE_MODEL` | `BAAI/bge-base-en-v1.5` | Dense embedding model |
| `SPLADE_ENCODER` | `naver/splade-cocondenser-ensembledistil` | SPLADE query encoder |

## Output

Results are saved to `results/`:

- **`summary.csv`** — one row per (dataset, method, trial) with NDCG, MAP, Recall, QPS, and index time
- **`raw/run_YYYYMMDD_HHMMSS.json`** — timestamped full results for reproducibility

## References

- Lin, J. et al. — *Operational Advice for Dense and Sparse Retrievers*
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [Pyserini](https://github.com/castorini/pyserini)
- [FAISS](https://github.com/facebookresearch/faiss)
````