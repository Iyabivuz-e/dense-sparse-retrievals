# from pyserini.search.lucene import LuceneSearcher
# import os
# import subprocess
# import time

# class BM25Retriever:
#     def __init__(self, index_path, top_k=1000):
#         self.index_path = index_path # path for the invented index
#         self.top_k = top_k
#         self.searcher = None
#         self.index_time = None
    
#     # This method builds the inverted index with the CLI from pyserini library
#     def index(self, input_path):
#         # input_path = the folder containing data
#         # index_path = the folder that will contain the inverted index
#         start = time.time()
#         result = subprocess.run(f"""
#         python -m pyserini.index.lucene \
#       --collection JsonCollection \
#       --input {input_path} \
#       --index {self.index_path} \
#       --generator DefaultLuceneDocumentGenerator \
#       --threads 4 \
#       --storePositions --storeDocvectors --storeRaw
#     """, shell=True, capture_output=True, text=True)
#         self.index_time = time.time() - start
        
#         print("STDOUT:", result.stdout[-2000:])  # last 2000 chars
#         print("STDERR:", result.stderr[-2000:])
#         print("Return code:", result.returncode)
            
#         if result.returncode != 0:
#             raise RuntimeError(f"Indexing failed with return code {result.returncode}")
        
#         self.searcher = LuceneSearcher(self.index_path)
        
#         return self.index_time
    
    
#     def search(self, queries, top_k):
#         if self.searcher is None:
#             raise RuntimeError("The index is not yet buit")
        
#         if isinstance(queries, str):
#             queries = [queries]
            
#         # all_results = []
#         start = time.time()

#         all_hits = [
#             self.searcher.search(query, top_k)
#             for query in queries
#         ]

#         end = time.time()
#         qps = len(queries) / (end - start)
        
#         all_results = []
#         for query_hits in all_hits:          # loop over each query's hits
#             results = []
#             for i, hit in enumerate(query_hits):   # loop over individual hits
#                 results.append({
#                     "rank": i + 1,
#                     "docid": hit.docid,
#                     "score": hit.score
#                 })
#             all_results.append(results) 
#         return all_results, qps


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
        all_hits = [self.searcher.search(q, top_k) for q in query_texts]
        end = time.time()

        qps = len(query_texts) / (end - start)

        # Convert to BEIR format: {qid: {docid: score}}
        results = {}
        for qid, hits in zip(query_ids, all_hits):
            results[qid] = {hit.docid: float(hit.score) for hit in hits}

        return results, qps