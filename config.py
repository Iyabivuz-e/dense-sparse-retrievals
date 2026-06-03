import os
import subprocess

#  Paths 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
INDEXES_DIR = os.path.join(ROOT_DIR, "indexes")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")
EMBEDDINGS_CACHE_DIR = os.path.join(ROOT_DIR, "embeddings_cache")


#  Environment (macOS JVM fix) 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["_JAVA_OPTIONS"] = "-Xmx8g" ## We give java 8GB of space
# we set JAVA dynamically find the path to Java 21 on macOS
try:
    java_home = subprocess.check_output(['/usr/libexec/java_home', '-v', '21']).decode('utf-8').strip()
    os.environ['JAVA_HOME'] = java_home
    print(f"JAVA_HOME successfully set to: {java_home}")
except subprocess.CalledProcessError:
    print("Java 21 not found on your system! Please run 'brew install openjdk@21' in your terminal.")
    
#  Datasets 
DATASETS = ["nfcorpus", "fiqa", "quora", "nq"]

# Methods 
METHODS = ["bm25", "splade", "flat", "flat_quantized", "hnsw", "hnsw_quantized"]

#  Hyperparameters 
TOP_K = 1000
HNSW_TRIALS = 5 #run multiple trials
HNSW_M = 16
HNSW_EF_SEARCH = 1000
HNSW_EF_CONSTRUCTION = 100
PQ_M = 8
PQ_NBITS = 8

#  Models 
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
SPLADE_ENCODER = "naver/splade-cocondenser-ensembledistil"