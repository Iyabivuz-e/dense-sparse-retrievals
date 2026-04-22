import faiss
# from models.bge import embed_model
import time
import json
from sentence_transformers import SentenceTransformer


class Dense:
    def __init__(self, model_name, documents_path, top_k=10):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.M = 16
        self.document_path = documents_path
        
        documents = []
        with open(documents_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc["text"])
            

        self.embeddings = self.model.encode(
            documents, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(self.embeddings)
        self.index = None
        self.index_time = None
        
class FlatIndex(Dense):
    
    """
    “We used cosine similarity via normalized embeddings, 
    which is standard practice for dense retrieval models, 
    while keeping the comparison between flat and HNSW consistent.”
    """
    def __init__(self, model_name, documents_path, top_k=10):
        super().__init__(model_name, documents_path, top_k)
        
    def build_index(self):
        dim = self.embeddings.shape[1]
        start_time = time.time()
        
        self.index = faiss.IndexFlatIP(dim)        
        self.index.add(self.embeddings)
        
        self.index_time = time.time() - start_time
        return self.index_time
        
    def search(self, queries=None, query_vectors=None):
        if self.index is None:
            raise ValueError("Index not built. Build it first.")
        
        if query_vectors is None:
            if isinstance(queries, str):
                queries = [queries]
                
            if not queries:
                return []
            query_vectors = self.model.encode(
                queries, convert_to_numpy=True
                ).astype("float32")
            faiss.normalize_L2(query_vectors)
            
        start = time.time()
        scores, indices = self.index.search(query_vectors, self.top_k)
        end = time.time()
        qps = len(query_vectors) / (end - start)
            
        results = []
        for q_id in range(len(query_vectors)):
            query_results = [
                {
                    "query_id": q_id,
                    "doc_id": i,
                    "score": scores[q_id][rank],
                    "rank": rank + 1,
                }
                for rank, i in enumerate(indices[q_id])
            ]
            results.append(query_results)
        return results, qps
          

class HNSW(Dense):
    def __init__(self, model_name, documents_path, top_k=10):
        super().__init__(model_name, documents_path, top_k)
        
    def build_index(self):
        dim = self.embeddings.shape[1]
        start_time = time.time()
        
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_INNER_PRODUCT)
        # Some configurations
        self.index.hnsw.efSearch = 100
        self.index.hnsw.efConstruction = 100
        
        self.index.add(self.embeddings)
        self.index_time = time.time() - start_time
        
    def search(self, queries):
        if self.index is None:
            raise ValueError("Index not built. Build it first.")
        
        if isinstance(queries, str):
            queries = [queries]
            
        if not queries:
            return []
        
        query_vectors = self.model.encode(
            queries, convert_to_numpy=True
            ).astype("float32")
        faiss.normalize_L2(query_vectors) ## Cosine similarity via normalized embeddings and inner product search”
        
        scores, indices = self.index.search(query_vectors, self.top_k)
        
        results = []
        for q_id in range(len(queries)):
            query_results = [
                {
                    "query_id": q_id,
                    "doc_id": i,
                    "score": scores[q_id][rank],
                    "rank": rank + 1,
                }
                for rank, i in enumerate(indices[q_id])
            ]
            results.append(query_results)
        return results

        
        
    