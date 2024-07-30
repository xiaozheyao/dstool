import numpy as np
import pandas as pd
from dstool.ml_utils.similarity import SimilarityIndexer

def calculate_task_similarity(args):
    print(args)
    df = pd.read_parquet(args.embedding_file)
    df[args.embedding_column] = df[args.embedding_column].apply(lambda x: np.stack(x, axis=0))
    tasks = df[args.task_column].values
    # compute task-task similarity
    similarities = []
    embedding_table = []
    for task in tasks:
        embedding_table.append(df[df[args.task_column]==task][args.embedding_column].values[0].flatten())
    embedding_table = np.array(embedding_table)
    similarity_indexer = SimilarityIndexer()
    similarity_indexer.create_index(embedding_table, 'angular', n_trees=args.n_trees)
    similarity_indexer.write_index(args.index_file)
    
    distances = similarity_indexer.get_all_distances()
    for i, task in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            similarities.append({
                "task1": task,
                "task2": task2,
                "distance": distances[i][j]
            })
    similarity_df = pd.DataFrame(similarities)
    # sort by similarity
    similarity_df = similarity_df.sort_values(by="distance", ascending=False)
    similarity_df.to_csv(args.output, index=False)
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task Similarity")
    parser.add_argument("--embedding-file", type=str, help="Parquet file containing task embeddings", required=True)
    parser.add_argument("--task-column", type=str, default="filename", help="Column name in the parquet file that contains the task name")
    parser.add_argument("--embedding-column", type=str, default="embeddings", help="Column name in the parquet file that contains the embeddings")
    parser.add_argument("--distance-metric", type=str, default="cosine", help="Distance metric to use for calculating similarity")
    parser.add_argument("--index-file", type=str, help="File to save the index", default="index.ann")
    parser.add_argument("--n-trees", type=int, help="Number of trees to use in the index", default=1024)
    parser.add_argument("--output", type=str, help="Output file to save the task similarity", default="task_similarity.csv")
    calculate_task_similarity(parser.parse_args())