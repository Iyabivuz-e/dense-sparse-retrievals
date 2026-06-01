def to_beir_format(raw_results, query_ids, doc_ids):
    """Convert retriever output to BEIR evaluation format.
    
    BEIR expects: {query_id: {doc_id: float_score, ...}, ...}
    """
    beir_results = {}
    for i, (qid, hits) in enumerate(zip(query_ids, raw_results)):
        beir_results[qid] = {}
        for hit in hits:
            docid = hit.get("docid") or doc_ids[hit["doc_id"]]
            beir_results[qid][docid] = float(hit["score"])
    return beir_results