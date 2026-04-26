
from sentence_transformers import SentenceTransformer
import faiss
import json

def embed_model(model_name, documents_path):
    model = SentenceTransformer(model_name)
    documents = []
    with open(documents_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc["text"])
            
    embeddings = model.encode(
        documents, convert_to_numpy=True
        ).astype("float32")
    faiss.normalize_L2(embeddings)
    
    return embeddings 

