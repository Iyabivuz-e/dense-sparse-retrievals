"""
BM25 retriever using Pyserini's Lucene backend.
Builds an inverted index via the Pyserini CLI, then searches with LuceneSearcher.
"""
import os
import json
import time
import subprocess

from pyserini.search.lucene import LuceneSearcher
from retrievers.base import BaseRetriever
import config


class BM25Retriever(BaseRetriever):

    def __init__(self):
        self.searcher = None

    def build_index(self, corpus: dict, dataset_name: str) -> float:
        base_dir = os.path.join(config.INDEXES_DIR, "bm25", dataset_name)
        corpus_dir = os.path.join(base_dir, "corpus")
        index_dir = os.path.join(base_dir, "index")
        os.makedirs(corpus_dir, exist_ok=True)

        # Convert corpus dict to corpus.jsonl 
        corpus_file = os.path.join(corpus_dir, "corpus.jsonl")
        with open(corpus_file, "w") as f:
            for doc_id, doc in corpus.items():
                record = {
                    "id": doc_id,
                    "contents": doc.get("title", "") + " " + doc.get("text", ""),
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Wrote {len(corpus)} docs → {corpus_file}")

        # Build Lucene inverted index
        start = time.time()
        result = subprocess.run(
            f"python -m pyserini.index.lucene "
            f"--collection JsonCollection "
            f"--input {corpus_dir} "
            f"--index {index_dir} "
            f"--generator DefaultLuceneDocumentGenerator "
            f"--threads 4 "
            f"--storePositions --storeDocvectors --storeRaw",
            shell=True,
            capture_output=True,
            text=True,
        )
        index_time = time.time() - start

        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:])
            print("STDERR:", result.stderr[-2000:])
            raise RuntimeError(
                f"BM25 indexing failed for {dataset_name} "
                f"(return code {result.returncode})"
            )

        # Initialize searcher 
        self.searcher = LuceneSearcher(index_dir)
        return index_time

    def search(self, queries: dict, top_k: int) -> tuple:
        if self.searcher is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        start = time.time()
        batch_results = self.searcher.batch_search(
            queries=query_texts,
            qids=query_ids,
            k=top_k,
            threads=4,

        )
        # all_hits = [self.searcher.search(q, top_k) for q in query_texts]
        end = time.time()

        qps = len(query_texts) / (end - start)

        # Convert to BEIR format: {qid: {docid: score}}
        results = {}
        for qid, hits in batch_results.items():
            results[qid] = {hit.docid: float(hit.score) for hit in hits}

        return results, qps