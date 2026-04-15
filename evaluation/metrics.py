## ndcg@10
from beir.retrieval.evaluation import EvaluateRetrieval

def compute_ndcg(qrels, results):
    retriever = EvaluateRetrieval()
    
    ndcg, map_score, recall, _ = retriever.evaluate(qrels, results, k_values=[1, 10, 100])
    
    return ndcg
