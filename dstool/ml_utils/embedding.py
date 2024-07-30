from typing import List

class EmbeddingGenerator():
    def __init__(self, model_name: str, pool=False):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("Please install sentence-transformers to use this class")
        if pool:
            self.pool = self.model.start_multi_process_pool()
        else:
            self.pool = None
            
    def encode(self, sentences: List[str]):
        if self.pool is not None:
            return self.model.encode_multi_process(sentences, self.pool)
        else:
            return self.model.encode(sentences)