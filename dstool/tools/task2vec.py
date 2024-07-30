import os
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from dstool.ml_utils.embedding import EmbeddingGenerator

def task2vec(args):
    print("Data Directory:", args.data_dir)
    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".jsonl")]
    data = []
    embedding_generator = EmbeddingGenerator(args.model_name, pool=False)
    for data_file in tqdm(files):
        filename = os.path.basename(data_file).split(".")[0]
        with open(data_file, "r") as f:
            sub_data = [json.loads(line) for line in f]
            # randomly sample `sample_size` number of tasks
            sub_data = np.random.choice(sub_data, args.sample_size, replace=True)
        embeddings = embedding_generator.encode([d[args.column_name] for d in sub_data]).tolist()
        data.append({
            "filename": filename,
            "embeddings": embeddings
        })
    df = pd.DataFrame(data)
    df.to_parquet(args.output)
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task2Vec")
    parser.add_argument("--data-dir", type=str, help="Directory to a bunch of `task_name.jsonl` files", required=True)
    parser.add_argument("--output", type=str, help="Directory to save the embeddings", required=True)
    parser.add_argument("--sample-size", type=int, help="Number of samples to use for training", default=256)
    parser.add_argument("--model-name", type=str, help="Name of the sentence-transformers model to use", default="all-MiniLM-L6-v2")
    parser.add_argument("--column-name", type=str, help="Column name in the jsonl file that contains the task description", default="text")
    task2vec(parser.parse_args())