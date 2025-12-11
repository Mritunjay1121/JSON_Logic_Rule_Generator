from sentence_transformers import SentenceTransformer
import numpy as np
import os
from loguru import logger


class EmbeddingService:
    """ Handles text embeddings using sentence transformers
    Pretty straightforward - just wraps the model"""
    
    def __init__(self, model_name=None):
        try:
            self.model_name = model_name or os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.success(f"Model loaded. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed_single(self, text):
        try:
            if not text or not text.strip():
                return np.zeros(self.dimension, dtype=np.float32)
            
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Single embedding failed: {str(e)}")
            return np.zeros(self.dimension, dtype=np.float32)
    
    def embed_batch(self, texts, batch_size=32):
        try:
            if not texts:
                return np.array([], dtype=np.float32)
            
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return np.zeros((len(texts), self.dimension), dtype=np.float32)
            
            # batch 
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # put embeddings back in right positions
            output = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, valid_idx in enumerate(valid_indices):
                output[valid_idx] = embeddings[i]
            
            return output.astype(np.float32)
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
    
    def cosine_similarity(self, vec1, vec2):
        # similarity between two vectors
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
        except Exception as e:
            logger.error(f"Cosine similarity failed: {str(e)}")
            return 0.0
    
    def batch_cosine_similarity(self, query_vec, corpus_vecs):
        # compare one query against many vectors
        # faster than looping through each one
        try:
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            
            # dot product = cosine sim for normalized vectors
            similarities = np.dot(corpus_vecs, query_norm)
            return np.clip(similarities, -1.0, 1.0)
        except Exception as e:
            logger.error(f"Batch cosine similarity failed: {str(e)}")
            return np.zeros(len(corpus_vecs))
    
    def get_dimension(self):
        """Get embedding dimension"""
        return self.dimension
