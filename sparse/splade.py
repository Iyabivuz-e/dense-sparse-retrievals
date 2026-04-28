from pyserini.search.lucene import LuceneImpactSearcher
import subprocess
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import json
import time

class Splade:
    def __init__(self, index_path, model, top_k=10):
        self.index_path=index_path
        self.top_k=top_k
        self.model_name=model
        self.searcher = None
        self.index_time = None
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model=AutoModelForMaskedLM.from_pretrained(model)
        self.model.eval()
                
    def tokenize_and_encode(self, text):
        
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        
        with torch.no_grad():
            outputs = self.model(**tokens).logits # Logits(probability distribution)
        
        ## Weights for the tokens
        weights = torch.log1p(torch.relu(outputs))
        # sparse_vec = torch.max(weights, dim=1).values
        
        ## Attention mask
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        weights = weights * attention_mask
        
        vec = torch.max(weights, dim=1).values.squeeze(0)
        vec = vec.cpu()
        
        return vec
        
    
    def convert_vec_to_dict(self, vec):
        indices = torch.nonzero(vec).squeeze(1)
        values = vec[indices]
        
        tokens_str = self.tokenizer.convert_ids_to_tokens(indices.tolist())
        vector = {
            token: float(value)
            for token, value in zip(tokens_str, values.tolist())
        }
        return vector
    

    def convert_corpus(self, input_file, output_file):
        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
            for line in f_in:
                doc = json.loads(line)

                vec = self.tokenize_and_encode(doc["contents"])
                vector = self.convert_vec_to_dict(vec)

                out = {
                    "id": doc["id"],
                    "vector": vector
                }

                f_out.write(json.dumps(out) + "\n")
        
        
    def index(self, input_path):
        start = time.time()
        results = subprocess.run(f"""
         python -m pyserini.index.lucene \
            --collection JsonVectorCollection \
            --input {input_path} \
            --index {self.index_path} \
            --generator DefaultLuceneDocumentGenerator \
            --threads 4 \
            --impact --pretokenized
        """, shell=True, capture_output=True, text=True)
        self.index_time = time.time() - start
        if results.returncode != 0:
            raise RuntimeError(f"Indexing failed with return code {results.returncode}")
        
        self.searcher = LuceneImpactSearcher(self.index_path, query_encoder=self.model_name)
        
        return self.index_time
    
    def search(self, queries):
        if self.searcher is None:
            raise RuntimeError("The index is not yet buit")
        
        if isinstance(queries, str):
            queries = [queries]
             
        
        start = time.time()
        hits = [self.searcher.search(query, k=self.top_k) for query in queries]
        end = time.time()
        
        qps = len(queries) / (end - start)
            
        all_results = []
        for hits in hits:
            results = [{"rank": i+1, "docid": hit.docid, "score": hit.score}
                    for i, hit in enumerate(hits)]
            all_results.append(results)
    
        return all_results, qps