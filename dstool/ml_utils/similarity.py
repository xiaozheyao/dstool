from numpy.typing import ArrayLike

class SimilarityIndexer():
    def __init__(self, filename=None):
        if filename is not None:
            raise NotImplementedError("Loading index from file is not implemented yet")
    
    def create_index(self, embeddings: ArrayLike, index_type: str, n_trees: int=10):
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise ImportError("Please install `annoy` to use this class")
        # embeddings are of shape (n_samples, n_features)
        if index_type not in ['angular', 'euclidean', 'manhattan', 'hamming', 'dot']:
            raise ValueError(f"Index type {index_type} not found")
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        self._index = AnnoyIndex(n_features, index_type)
        for i, emb in enumerate(embeddings):
            self._index.add_item(i, emb)
        self.n_samples = n_samples
        self.n_features = n_features
        print(f'Building index with {n_trees} trees')
        self._index.build(n_trees, n_jobs=-1)
        
    def get_all_distances(self):
        distances = {}
        for i in range(self.n_samples):
            distances[i] = {}
            for j in range(self.n_samples):
                distances[i][j] = self._index.get_distance(i, j)
        return distances

    def write_index(self, filename: str):
        self._index.save(filename)