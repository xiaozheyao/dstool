from typing import List, Optional

class TextEncoder:
    def __init__(self, model_name: str="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers to use TextEncoder.")
        self.model = SentenceTransformer(model_name)

    def compute_simlarity(self, texts: List[str]):
        embeddings = self.compute_embedding(texts)
        similarities = self.model.similarity(embeddings, embeddings)
        return similarities

    def compute_embedding(self, texts: List[str]):
        return self.model.encode(texts)