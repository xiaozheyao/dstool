import os
import json
from dstool.ml_utils.embedding import EmbeddingGenerator

def task2vec(args):
    print("Data Directory:", args.data_directory)
    files = [os.path.join(args.data_directory, f) for f in os.listdir(args.data_directory) if f.endswith(".jsonl")]
    data = []
    for data_file in files:
        with open(data_file, "r") as f:
            data.extend([json.loads(line) for line in f])
    print("Number of tasks:", len(data))
    print("Sample Size:", args.sample_size)
    embedding_generator = EmbeddingGenerator(args.model_name, pool=True)
    
    
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task2Vec")
    parser.add_argument("--data-directory", type=str, help="Directory to a bunch of `task_name.jsonl` files", required=True)
    parser.add_argument("--output-directory", type=str, help="Directory to save the embeddings", required=True)
    parser.add_argument("--sample-size", type=int, help="Number of samples to use for training", default=256)
    parser.add_argument("--model-name", type=str, help="Name of the sentence-transformers model to use", default="all-MiniLM-L6-v2")
    parser.add_argument("--column-name", type=str, help="Column name in the jsonl file that contains the task description", default="text")
    task2vec(parser.parse_args())