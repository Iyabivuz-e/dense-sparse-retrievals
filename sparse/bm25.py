from pyserini.search.lucene import LuceneSearcher
import os
import subprocess

class BM25Retrieval:
    def __init__(self, index_path, top_k=10):
        self.index_path = index_path # path for the invented index
        self.top_k = top_k
        self.searcher = None
    
    # This method builds the inverted index with the CLI from pyserini library
    def index(self, input_path):
        # input_path = the folder containing data
        # index_path = the folder that will contain the inverted index
        result = subprocess.run(f"""
        python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input {input_path} \
      --index {self.index_path} \
      --generator DefaultLuceneDocumentGenerator \
      --threads 4 \
      --storePositions --storeDocvectors --storeRaw
    """, shell=True, capture_output=True, text=True)
        
    
        if result.returncode != 0:
            raise RuntimeError(f"Indexing failed with return code {result.returncode}")

        
        self.searcher = LuceneSearcher(self.index_path)
        
    
    
    def search(self, queries):
        if self.searcher is None:
            raise RuntimeError("The index is not yet buit")
        
        if isinstance(queries, str):
            queries = [queries]
            
        all_results = []
        for query in queries:
            hits = self.searcher.search(query, self.top_k)
            results = []
            for i, hit in enumerate(hits):
                results.append({
                "rank": i+1,
                "docid": hit.docid,
                "score": hit.score
                })
            all_results.append(results)
        return all_results