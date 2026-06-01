# from pyserini.search.lucene import LuceneImpactSearcher
# import subprocess
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import torch
# import json
# import time
# import tqdm


# class Splade:
#     def __init__(self, index_path, model, top_k=10):
#         self.index_path=index_path
#         self.top_k=top_k
#         self.model_name=model
#         self.searcher = None
#         self.index_time = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.tokenizer = AutoTokenizer.from_pretrained(model)
#         self.model=AutoModelForMaskedLM.from_pretrained(model)
#         self.model.to(self.device)
#         self.model.eval()
                
#     def tokenize_and_encode(self, text):
        
#         tokens = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#         )
#         tokens = {k: v.to(self.device) for k, v in tokens.items()}
#         with torch.no_grad():
#             outputs = self.model(**tokens).logits # Logits(probability distribution)
        
#         ## Weights for the tokens
#         weights = torch.log1p(torch.relu(outputs))
#         # sparse_vec = torch.max(weights, dim=1).values
        
#         ## Attention mask
#         attention_mask = tokens["attention_mask"].unsqueeze(-1)
#         weights = weights * attention_mask
        
#         vec = torch.max(weights, dim=1).values.squeeze(0)
#         vec = vec.cpu()
        
#         return vec
        
    
#     def convert_vec_to_dict(self, vec):
#         indices = torch.nonzero(vec).squeeze(1)
#         values = vec[indices]
        
#         tokens_str = self.tokenizer.convert_ids_to_tokens(indices.tolist())
#         vector = {
#             token: float(value)
#             for token, value in zip(tokens_str, values.tolist())
#         }
#         return vector
    

#     # def convert_corpus(self, input_file, output_file):
#     #     with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
#     #         for line in f_in:
#     #             doc = json.loads(line)

#     #             vec = self.tokenize_and_encode(doc["contents"])
#     #             vector = self.convert_vec_to_dict(vec)

#     #             out = {
#     #                 "id": doc["id"],
#     #                 "vector": vector
#     #             }

#     #             f_out.write(json.dumps(out) + "\n")
    
#     def convert_corpus(self, input_file, output_file, batch_size=32):
        
#         # Step 1: Read ALL docs into RAM first (avoid repeated Drive I/O)
#         print("Reading corpus into memory...")
#         docs = []
#         with open(input_file, "r") as f_in:
#             for line in f_in:
#                 docs.append(json.loads(line))
#         print(f"Loaded {len(docs)} documents")
        
#         # Step 2: Encode in batches with progress bar
#         with open(output_file, "w") as f_out:
#             for i in tqdm(range(0, len(docs), batch_size), desc="Encoding"):
#                 batch = docs[i : i + batch_size]
#                 texts = [doc["contents"] for doc in batch]
                
#                 tokens = self.tokenizer(
#                     texts,
#                     return_tensors="pt",
#                     truncation=True,
#                     padding=True,
#                     max_length=512,
#                 )
#                 tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
#                 with torch.no_grad():
#                     outputs = self.model(**tokens).logits
                
#                 weights = torch.log1p(torch.relu(outputs))
#                 attention_mask = tokens["attention_mask"].unsqueeze(-1)
#                 weights = weights * attention_mask
#                 vecs = torch.max(weights, dim=1).values.cpu()
                
#                 for doc, vec in zip(batch, vecs):
#                     vector = self.convert_vec_to_dict(vec)
#                     f_out.write(json.dumps({"id": doc["id"], "vector": vector}) + "\n")

        
#     def index(self, input_path):
#         start = time.time()
#         results = subprocess.run(f"""
#          python -m pyserini.index.lucene \
#             --collection JsonVectorCollection \
#             --input {input_path} \
#             --index {self.index_path} \
#             --generator DefaultLuceneDocumentGenerator \
#             --threads 4 \
#             --impact --pretokenized
#         """, shell=True, capture_output=True, text=True)
#         self.index_time = time.time() - start
#         if results.returncode != 0:
#             raise RuntimeError(f"Indexing failed with return code {results.returncode}")
        
#         self.searcher = LuceneImpactSearcher(self.index_path, query_encoder=self.model_name)
        
#         return self.index_time
    
#     def search(self, queries):
#         if self.searcher is None:
#             raise RuntimeError("The index is not yet buit")
        
#         if isinstance(queries, str):
#             queries = [queries]
             
        
#         start = time.time()
#         hits = [self.searcher.search(query, k=self.top_k) for query in queries]
#         end = time.time()
        
#         qps = len(queries) / (end - start)
            
#         all_results = []
#         for hits in hits:
#             results = [{"rank": i+1, "docid": hit.docid, "score": hit.score}
#                     for i, hit in enumerate(hits)]
#             all_results.append(results)
    
#         return all_results, qps


       
# import time
# from pyserini.search.lucene import LuceneImpactSearcher

# class SPLADE:
#     def __init__(self, dataset: str, query_encoder: str):
#     #     self.dataset = dataset
#     #     self.query_encoder = query_encoder

#     #     index_name = f"beir-v1.0.0-{dataset}.splade-pp-ed"
#     #     self.searcher = LuceneImpactSearcher.from_prebuilt_index(
#     #         index_name,
#     #         query_encoder=query_encoder,
#     #     )def __init__(self, dataset: str, query_encoder: str):
#         self.dataset = dataset
#         self.query_encoder = query_encoder
#         index_name = f"beir-v1.0.0-{dataset}.splade-pp-ed"
        
#         # 1. Dynamically register the index in Pyserini's index database at runtime if missing
#         from pyserini.prebuilt_index_info import IMPACT_INDEX_INFO
#         if index_name not in IMPACT_INDEX_INFO:
#             IMPACT_INDEX_INFO[index_name] = {
#                 "description": f"Anserini Lucene impact index of BEIR collection '{dataset}' encoded by SPLADE++ CoCondenser-EnsembleDistil",
#                 "filename": f"lucene-inverted.beir-v1.0.0-{dataset}.splade-pp-ed.20231124.a66f86f.tar.gz",
#                 "urls": [
#                     f"https://huggingface.co/datasets/castorini/prebuilt-indexes-beir/resolve/main/lucene-inverted/splade-pp-ed/lucene-inverted.beir-v1.0.0-{dataset}.splade-pp-ed.20231124.a66f86f.tar.gz"
#                 ],
#                 "md5": None,  # Setting md5 to None bypasses strict validation and works dynamically for any BEIR dataset
#             }
#         # 2. Initialize the searcher (Pyserini will auto-download and cache if it's the first run)
#         self.searcher = LuceneImpactSearcher.from_prebuilt_index(
#             index_name,
#             query_encoder=query_encoder,
#         )
    

#     def search(self, queries, top_k: int = 1000):
#         start = time.time()

#         all_hits = [
#             self.searcher.search(query, k=top_k)
#             for query in queries
#         ]

#         end = time.time()
#         qps = len(queries) / (end - start)

#         all_results = []
#         for query_hits in all_hits:
#             results = [
#                 {
#                     "rank": i + 1,
#                     "docid": hit.docid,
#                     "score": hit.score,
#                 }
#                 for i, hit in enumerate(query_hits)
#             ]
#             all_results.append(results)

#         return all_results, qps


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
        all_hits = [self.searcher.search(q, k=top_k) for q in query_texts]
        end = time.time()

        qps = len(query_texts) / (end - start)

        # Convert to BEIR format
        results = {}
        for qid, hits in zip(query_ids, all_hits):
            results[qid] = {hit.docid: float(hit.score) for hit in hits}

        return results, qps