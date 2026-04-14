import os
import json

def convert_corpus_jsonl(corpus, output_path):
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path,  "corpus.jsonl")
    with open(output_file, "w") as f:
        for doc_id, doc_data in corpus.items():
            record = {
                "id": doc_id,
                "contents": doc_data.get("title", "") + " " + doc_data.get("text", "")
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(corpus)} documents to {output_file}")
    return output_path