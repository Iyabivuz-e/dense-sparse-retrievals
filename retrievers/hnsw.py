"""
HNSW dense retriever with FAISS IndexHNSWFlat.
Optionally uses Product Quantization (IndexHNSWPQ) for compression.

Since HNSW graph construction is NON-DETERMINISTIC we run multiple
trials(5) and averaging.
"""
import time
import faiss
from retrievers.base import BaseRetriever
from models.embeddings import get_embeddings, get_model
import config


class HNSWRetriever(BaseRetriever):

    def __init__(self, quantize: bool = False):
        self.quantize = quantize
        self.index = None
        self.doc_ids = None

    def build_index(self, corpus: dict, dataset_name: str) -> float:
        # Get embeddings cached
        embeddings, self.doc_ids = get_embeddings(
            corpus,
            config.DENSE_MODEL,
            dataset_name,
            cache_dir=config.EMBEDDINGS_CACHE_DIR,
        )

        dim = embeddings.shape[1]

        # Build HNSW graph
        start = time.time()

        if self.quantize:
            self.index = faiss.IndexHNSWPQ(
                dim,
                config.HNSW_M,
                config.PQ_M,
                config.PQ_NBITS,
                faiss.METRIC_INNER_PRODUCT,
            )
            self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
            self.index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
            self.index.train(embeddings)
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexHNSWFlat(
                dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
            self.index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
            self.index.add(embeddings)

        index_time = time.time() - start
        return index_time

    def search(self, queries: dict, top_k: int) -> tuple:
        if self.index is None:
            raise RuntimeError("Index not built. Build it first.")

        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        # Encode queries
        model = get_model(config.DENSE_MODEL)
        query_vectors = model.encode(
            query_texts, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(query_vectors) ## Cosine similarity via normalized embeddings and inner product search”

        #  FAISS search
        start = time.time()
        scores, indices = self.index.search(query_vectors, top_k)
        end = time.time()

        qps = len(query_texts) / (end - start)

        # Convert to BEIR format
        results = {}
        for i, qid in enumerate(query_ids):
            results[qid] = {}
            for rank in range(len(indices[i])):
                doc_idx = indices[i][rank]
                if doc_idx == -1:
                    continue
                doc_id = self.doc_ids[doc_idx]
                results[qid][doc_id] = float(scores[i][rank])

        return results, qps