import os
from beir import util
# from beir.datasets.data_loader import GenericDataLoader

def download_dataset():
    dataset = "nfcorpus"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    output_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url=url, out_dir=output_dir)
    print("the dataset is downloaded sucessfully here", data_path)
    

# def load_data():
#     data_path = "datasets/nfcorpus"
#     corpus, query, qrels = GenericDataLoader(data_path).load_custom()
    
#     return corpus, query, qrels
    