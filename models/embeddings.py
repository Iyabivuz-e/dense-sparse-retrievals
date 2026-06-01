
# from sentence_transformers import SentenceTransformer
# import faiss
# import json

# def embed_model(model_name, documents_path):
#     model = SentenceTransformer(model_name)
#     documents = []
#     with open(documents_path, 'r') as f:
#         for line in f:
#             doc = json.loads(line)
#             documents.append(doc["contents"])
            
#     embeddings = model.encode(
#         documents, convert_to_numpy=True
#         ).astype("float32")
#     faiss.normalize_L2(embeddings)
    
#     return embeddings 
# # 


"""
Shared embedding utilities for dense retrievers.

Key feature: **caching**.  The first dense retriever to run on a dataset
computes embeddings and saves them to disk + memory.  Subsequent
retrievers (e.g. HNSW after Flat) reuse the cache — zero redundant work.
"""
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# In-memory caches
_model_cache = {}
_embedding_cache = {}


def get_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model (cached in memory after first load)."""
    if model_name not in _model_cache:
        print(f"  Loading model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def get_embeddings(
    corpus: dict,
    model_name: str,
    dataset_name: str,
    cache_dir: str = None,
) -> tuple:
    """Compute or load cached corpus embeddings.

    Caching order:
        1. In-memory dict  (instant — same Python process)
        2. Disk .npy files  (fast — avoids re-encoding)
        3. Compute from scratch  (slow — encodes all docs)
    """
    cache_key = (dataset_name, model_name)

    # In-memory cache 
    if cache_key in _embedding_cache:
        print(f"  In-memory cached embeddings for {dataset_name}")
        return _embedding_cache[cache_key]

    # Disk cache
    if cache_dir:
        emb_path = os.path.join(cache_dir, dataset_name, "embeddings.npy")
        ids_path = os.path.join(cache_dir, dataset_name, "doc_ids.npy")
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            print(f"  Loading embeddings from disk for {dataset_name}")
            embeddings = np.load(emb_path)
            doc_ids = np.load(ids_path, allow_pickle=True).tolist()
            _embedding_cache[cache_key] = (embeddings, doc_ids)
            return embeddings, doc_ids

    # Compute from scratch
    print(f" Computing embeddings for {dataset_name} ({len(corpus)} docs)...")
    model = get_model(model_name)

    doc_ids = list(corpus.keys())
    texts = [
        corpus[did].get("title", "") + " " + corpus[did].get("text", "")
        for did in doc_ids
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=256,
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    # Save to disk
    if cache_dir:
        save_dir = os.path.join(cache_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
        np.save(
            os.path.join(save_dir, "doc_ids.npy"),
            np.array(doc_ids, dtype=object),
        )
        print(f" Saved embeddings → {save_dir}")

    _embedding_cache[cache_key] = (embeddings, doc_ids)
    return embeddings, doc_ids