# import os
# from beir import util
# # from beir.datasets.data_loader import GenericDataLoader

# def download_dataset():
#     dataset = "fiqa"
#     url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
#     output_dir = os.path.join(os.getcwd(), "datasets")
#     data_path = util.download_and_unzip(url=url, out_dir=output_dir)
#     print("the dataset is downloaded sucessfully here", data_path)
    

# # def load_data():
# #     data_path = "datasets/nfcorpus"
# #     corpus, query, qrels = GenericDataLoader(data_path).load_custom()
    
# #     return corpus, query, qrels
    
    
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os

def load_dataset(dataset_name: str, base_dir: str) -> tuple:
    """Download and load a BEIR dataset.
    Returns (corpus, queries, qrels)."""
    
    data_path = os.path.join(base_dir, dataset_name)
    
    if not os.path.exists(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url=url, out_dir=base_dir)
        print(f"  Downloaded {dataset_name} → {data_path}")
    
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return corpus, queries, qrels