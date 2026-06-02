"""
SPLADE++ retriever using Pyserini's prebuilt impact indexes.
"""
import time
from pyserini.search.lucene import LuceneImpactSearcher
from retrievers.base import BaseRetriever
import config


class SPLADERetriever(BaseRetriever):

    def __init__(self):
        self.searcher = None

    def build_index(self, corpus: dict, dataset_name: str) -> float:
        """Load and auto-download the prebuilt SPLADE index."""
        index_name = f"beir-v1.0.0-{dataset_name}.splade-pp-ed"

        #  Dynamic registration for Pyserini 0.22.1 due to Older versions don't have BEIR indexes in their registry.
        from pyserini.prebuilt_index_info import IMPACT_INDEX_INFO

        if index_name not in IMPACT_INDEX_INFO:
            IMPACT_INDEX_INFO[index_name] = {
                "description": (
                    f"Lucene impact index of BEIR '{dataset_name}' "
                    f"encoded by SPLADE++ CoCondenser-EnsembleDistil"
                ),
                "filename": (
                    f"lucene-inverted.beir-v1.0.0-{dataset_name}"
                    f".splade-pp-ed.20231124.a66f86f.tar.gz"
                ),
                "urls": [
                    f"https://huggingface.co/datasets/castorini/"
                    f"prebuilt-indexes-beir/resolve/main/"
                    f"lucene-inverted/splade-pp-ed/"
                    f"lucene-inverted.beir-v1.0.0-{dataset_name}"
                    f".splade-pp-ed.20231124.a66f86f.tar.gz"
                ],
                "md5": None,   # Bypasses strict checksum validation
            }
            print(f"  Registered index metadata: {index_name}")

        # we then initialize searcher which downloads the index on first run
        start = time.time()
        self.searcher = LuceneImpactSearcher.from_prebuilt_index(
            index_name,
            query_encoder=config.SPLADE_ENCODER,
        )
        index_time = time.time() - start

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
            threads=4
        )
        # all_hits = [self.searcher.search(q, k=top_k) for q in query_texts]
        end = time.time()

        qps = len(query_texts) / (end - start)

        # Convert to BEIR format
        results = {}
        for qid, hits in batch_results.items():
            results[qid] = {hit.docid: float(hit.score) for hit in hits}

        return results, qps