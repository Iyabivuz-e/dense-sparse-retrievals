# """
# HNSW dense retriever with FAISS IndexHNSWFlat.
# Optionally uses Product Quantization (IndexHNSWPQ) for compression.

# Since HNSW graph construction is NON-DETERMINISTIC we run multiple
# trials(5) and averaging.
# """
# import time
# import faiss
# from retrievers.base import BaseRetriever
# from models.embeddings import get_embeddings, get_model
# import config


# class HNSWRetriever(BaseRetriever):

#     def __init__(self, quantize: bool = False):
#         self.quantize = quantize
#         self.index = None
#         self.doc_ids = None

#     def build_index(self, corpus: dict, dataset_name: str) -> float:
#         # Get embeddings cached
#         embeddings, self.doc_ids = get_embeddings(
#             corpus,
#             config.DENSE_MODEL,
#             dataset_name,
#             cache_dir=config.EMBEDDINGS_CACHE_DIR,
#         )

#         dim = embeddings.shape[1]

#         # Build HNSW graph
#         start = time.time()

#         if self.quantize:
#             self.index = faiss.IndexHNSWPQ(
#                 dim,
#                 config.HNSW_M,
#                 config.PQ_M,
#                 config.PQ_NBITS,
#                 faiss.METRIC_INNER_PRODUCT,
#             )
#             self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
#             self.index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
#             self.index.train(embeddings)
#             self.index.add(embeddings)
#         else:
#             self.index = faiss.IndexHNSWFlat(
#                 dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT
#             )
#             self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
#             self.index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
#             self.index.add(embeddings)

#         index_time = time.time() - start
#         return index_time

#     def search(self, queries: dict, top_k: int) -> tuple:
#         if self.index is None:
#             raise RuntimeError("Index not built. Build it first.")

#         query_ids = list(queries.keys())
#         query_texts = list(queries.values())

#         # Encode queries
#         model = get_model(config.DENSE_MODEL)
#         query_vectors = model.encode(
#             query_texts, convert_to_numpy=True
#         ).astype("float32")
#         faiss.normalize_L2(query_vectors) ## Cosine similarity via normalized embeddings and inner product search”

#         #  FAISS search
#         start = time.time()
#         scores, indices = self.index.search(query_vectors, top_k)
#         end = time.time()

#         qps = len(query_texts) / (end - start)

#         # Convert to BEIR format
#         results = {}
#         for i, qid in enumerate(query_ids):
#             results[qid] = {}
#             for rank in range(len(indices[i])):
#                 doc_idx = indices[i][rank]
#                 if doc_idx == -1:
#                     continue
#                 doc_id = self.doc_ids[doc_idx]
#                 results[qid][doc_id] = float(scores[i][rank])

#         return results, qps



"""
HNSW dense retriever using Lucene's native HNSW vector index.
"""
import os
import json
import time
import subprocess


from pyserini.search.lucene import LuceneHnswDenseSearcher
from pyserini.pyclass import autoclass
from retrievers.base import BaseRetriever
import config


class HNSWRetriever(BaseRetriever):

    def __init__(self, quantize: bool = False):
        self.searcher = None
        self.quantize = quantize

    # def _encode_corpus(self, corpus: dict, dataset_name: str) -> str:
    #     """Encode corpus documents into JSONL with dense vectors.
    #     Reuses cached vectors if they already exist
    #     """
    #     base_dir = os.path.join(config.INDEXES_DIR, "dense", dataset_name)
    #     corpus_dir = os.path.join(base_dir, "corpus")
    #     vectors_dir = os.path.join(base_dir, "vectors")
    #     os.makedirs(corpus_dir, exist_ok=True)
    #     os.makedirs(vectors_dir, exist_ok=True)

    #     # Write corpus as JSONL
    #     corpus_file = os.path.join(corpus_dir, "corpus.jsonl")
    #     if not os.path.exists(corpus_file):
    #         with open(corpus_file, "w") as f:
    #             for doc_id, doc in corpus.items():
    #                 record = {
    #                     "id": doc_id,
    #                     "text": doc.get("title", "") + " " + doc.get("text", ""),
    #                 }
    #                 f.write(json.dumps(record) + "\n")
    #         print(f"  Wrote {len(corpus)} docs → {corpus_file}")

    #     # Encode (cached)
    #     vectors_file = os.path.join(vectors_dir, "embeddings.jsonl")
    #     if not os.path.exists(vectors_file):
    #         print(f"  Encoding {len(corpus)} docs with {config.DENSE_MODEL}...")
    #         result = subprocess.run(
    #             f"python -m pyserini.encode "
    #             f"  input  --corpus {corpus_dir} --fields text "
    #             f"  output --embeddings {vectors_dir} "
    #             f"  encoder --encoder {config.DENSE_MODEL} "
    #             f"         --encoder-class auto "
    #             f"         --fields text "
    #             f"         --batch-size 256 "
    #             f"         --l2-norm "
    #             f"         --device cpu",
    #             shell=True,
    #             capture_output=True,
    #             text=True,
    #         )
    #         if result.returncode != 0:
    #             print("STDERR:", result.stderr[-2000:])
    #             raise RuntimeError(f"Encoding failed for {dataset_name}")
    #         print(f"  Encoded vectors → {vectors_dir}")
    #     else:
    #         print(f"  Using cached vectors from {vectors_dir}")

    #     return vectors_dir
    
    def _encode_corpus(self, corpus: dict, dataset_name: str) -> str:
        """Encode corpus documents and save as JSONL vectors
        """
        base_dir = os.path.join(config.INDEXES_DIR, "dense", dataset_name)
        vectors_dir = os.path.join(base_dir, "vectors")
        os.makedirs(vectors_dir, exist_ok=True)

        vectors_file = os.path.join(vectors_dir, "embeddings.jsonl")

        if os.path.exists(vectors_file):
            print(f"  Using cached vectors from {vectors_dir}")
            return vectors_dir

        # Encode in-process with sentence-transformers
        from sentence_transformers import SentenceTransformer
        import numpy as np

        print(f"  Encoding {len(corpus)} docs with {config.DENSE_MODEL}...")
        model = SentenceTransformer(config.DENSE_MODEL)

        doc_ids = list(corpus.keys())
        texts = [
            corpus[did].get("title", "") + " " + corpus[did].get("text", "")
            for did in doc_ids
        ]

        embeddings = model.encode(
            texts, convert_to_numpy=True,
            show_progress_bar=True, batch_size=256,
        ).astype("float32")

        # L2 normalize using cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Write in JsonDenseVectorCollection format
        with open(vectors_file, "w") as f:
            for doc_id, vec in zip(doc_ids, embeddings):
                record = {"id": doc_id, "vector": vec.tolist()}
                f.write(json.dumps(record) + "\n")

        print(f"  Saved {len(doc_ids)} vectors::  {vectors_file}")
        return vectors_dir

    def build_index(self, corpus: dict, dataset_name: str) -> float:
        base_dir = os.path.join(config.INDEXES_DIR, "dense", dataset_name)
        index_dir = os.path.join(base_dir, "hnsw-index")

        # Encode corpus (cached)
        vectors_dir = self._encode_corpus(corpus, dataset_name)
        
        quantize_flag = "-quantize int8" if self.quantize else ""
        
        JIndexHnswDenseVectors = autoclass('io.anserini.index.IndexHnswDenseVectors')

        # Build Lucene HNSW index 
        # print(f"  Building Lucene HNSW index...")
        # start = time.time()
        # result = subprocess.run(
        #     f"python -m pyserini.index.lucene "
        #     f"--collection JsonDenseVectorCollection "
        #     f"--input {vectors_dir} "
        #     f"--index {index_dir} "
        #     f"--generator DenseVectorDocumentGenerator "
        #     f"--threads 4 "
        #     f"-indexType hnsw {quantize_flag}"
        #     f"-M {config.HNSW_M} "
        #     f"-efC {config.HNSW_EF_CONSTRUCTION}",
        #     shell=True,
        #     capture_output=True,
        #     text=True,
        # )
        # index_time = time.time() - start

        # if result.returncode != 0:
        #     print("STDOUT:", result.stdout[-2000:])
        #     print("STDERR:", result.stderr[-2000:])
        #     raise RuntimeError(f"HNSW indexing failed for {dataset_name}")
        
        print(f"  Building Lucene HNSW index (M={config.HNSW_M}, efC={config.HNSW_EF_CONSTRUCTION})...")
        start = time.time()
        args = [
            '-input', vectors_dir,
            '-index', index_dir,
            '-collection', 'JsonDenseVectorCollection',
            '-generator', 'DenseVectorDocumentGenerator',
            '-threads', '4',
            '-M', str(config.HNSW_M),
            '-efC', str(config.HNSW_EF_CONSTRUCTION),
        ]
        JIndexHnswDenseVectors.main(args)
        index_time = time.time() - start


        # Initialize searcher
        self.searcher = LuceneHnswDenseSearcher(
            index_dir,
            ef_search=config.HNSW_EF_SEARCH,
            encoder="BgeBaseEn15",
        )
        return index_time

    def search(self, queries: dict, top_k: int) -> tuple:
        if self.searcher is None:
            raise RuntimeError("Index not built. Build it first.")

        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        start = time.time()
        batch_results = self.searcher.batch_search(
            queries=query_texts,
            qids=query_ids,
            k=top_k,
            threads=4,
        )
        end = time.time()

        qps = len(query_texts) / (end - start)

        results = {}
        for qid, hits in batch_results.items():
            results[qid] = {hit.docid: float(hit.score) for hit in hits}

        return results, qps