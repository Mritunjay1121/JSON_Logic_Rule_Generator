import numpy as np
from rank_bm25 import BM25Okapi
import re
import os
from loguru import logger
from app.constants import SAMPLE_STORE_KEYS, build_key_search_text
from app.models import KeyMapping
from app.services.embedding_service import EmbeddingService


class KeyMapper:
    """ Hybrid approach - combines semantic search with keyword matching
    TODO: maybe add cross-encoder reranking later if needed """
    
    def __init__(self, embedding_service):
        try:
            self.embed_service = embedding_service
            self.rrf_k = int(os.getenv("RRF_K", "60"))  # k=60 worked best in testing
            self.threshold = float(os.getenv("SIM_THRESHOLD", "0.7"))
            
            logger.info("Initializing KeyMapper...")
            
            self.keys = SAMPLE_STORE_KEYS
            
            # build text for each key to search against
            self.key_texts = [build_key_search_text(k) for k in self.keys]
            logger.debug(f"Built {len(self.key_texts)} key search texts")
            
            # precompute embeddings so we dont have to do it every time
            logger.info("Computing key embeddings...")
            self.key_embeddings = self.embed_service.embed_batch(self.key_texts)
            logger.debug(f"Key embeddings shape: {self.key_embeddings.shape}")
            
            # setup BM25 for keyword matching
            logger.info("Building BM25 index...")
            self.tokenized_keys = [self.tokenize(text) for text in self.key_texts]
            self.bm25 = BM25Okapi(self.tokenized_keys)
            logger.success("KeyMapper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KeyMapper: {str(e)}")
            raise
    
    def tokenize(self, text):
        # simple tokenization - just split on word boundaries
        try:
            tokens = re.findall(r'\w+', text.lower())
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            return []
    
    def extract_key_phrases(self, prompt):
        # extract different phrase combinations from prompt
        # helps match to specific parts of the prompt
        try:
            phrases = []
            
            phrases.append(prompt.strip())
            
            tokens = self.tokenize(prompt)
            
            # bigrams - pairs of words
            for i in range(len(tokens) - 1):
                phrases.append(f"{tokens[i]} {tokens[i+1]}")
            
            # trigrams - three word combos
            for i in range(len(tokens) - 2):
                phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
            
            # add longer tokens only (skip short words like 'is', 'or')
            phrases.extend([t for t in tokens if len(t) > 3])
            
            # remove dupes but keep order
            seen = set()
            unique = []
            for p in phrases:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            
            return unique[:15]  # limit to avoid too many
        except Exception as e:
            logger.error(f"Phrase extraction failed: {str(e)}")
            return [prompt]  # fallback to just the prompt
    
    def compute_dense_ranks(self, prompt):
        # get semantic similarity using embeddings
        try:
            prompt_emb = self.embed_service.embed_single(prompt)
            
            similarities = self.embed_service.batch_cosine_similarity(
                prompt_emb, 
                self.key_embeddings
            )
            
            # sort by similarity
            ranks = np.argsort(-similarities)
            
            # convert to rank positions starting from 1
            rank_positions = np.zeros(len(self.keys), dtype=int)
            for pos, idx in enumerate(ranks):
                rank_positions[idx] = pos + 1
            
            return rank_positions, similarities
        except Exception as e:
            logger.error(f"Dense ranking failed: {str(e)}")
            # return default ranks if something breaks
            default_ranks = np.arange(1, len(self.keys) + 1)
            default_sims = np.zeros(len(self.keys))
            return default_ranks, default_sims
    
    def compute_sparse_ranks(self, prompt):
        # keyword-based matching with BM25
        try:
            prompt_tokens = self.tokenize(prompt)
            bm25_scores = self.bm25.get_scores(prompt_tokens)
            
            ranks = np.argsort(-bm25_scores)
            
            rank_positions = np.zeros(len(self.keys), dtype=int)
            for pos, idx in enumerate(ranks):
                rank_positions[idx] = pos + 1
            
            return rank_positions, bm25_scores
        except Exception as e:
            logger.error(f"Sparse ranking failed: {str(e)}")
            default_ranks = np.arange(1, len(self.keys) + 1)
            default_scores = np.zeros(len(self.keys))
            return default_ranks, default_scores
    
    def apply_rrf(self, dense_ranks, sparse_ranks):
        # reciprocal rank fusion - combines both ranking methods
        # formula from research paper, works better than weighted average
        try:
            rrf_scores = (1.0 / (self.rrf_k + dense_ranks)) + \
                         (1.0 / (self.rrf_k + sparse_ranks))
            return rrf_scores
        except Exception as e:
            logger.error(f"RRF fusion failed: {str(e)}")
            # fallback to just dense ranks
            return 1.0 / (self.rrf_k + dense_ranks)
    
    def map_keys(self, prompt, top_k=5):
        """Map user prompt to actual store keys"""
        try:
            logger.info(f"Mapping keys for prompt: {prompt[:50]}...")
            
            # get rankings from both methods
            dense_ranks, dense_sims = self.compute_dense_ranks(prompt)
            sparse_ranks, sparse_scores = self.compute_sparse_ranks(prompt)
            
            # combine them using RRF
            rrf_scores = self.apply_rrf(dense_ranks, sparse_ranks)
            
            # sort by combined score
            sorted_indices = np.argsort(-rrf_scores)
            
            # extract phrases from prompt
            key_phrases = self.extract_key_phrases(prompt)
            
            # build the mappings
            mappings = []
            for idx in sorted_indices:
                # normalize score to 0-1 range
                max_rrf = 2.0 / (self.rrf_k + 1)
                normalized_score = float(rrf_scores[idx] / max_rrf)
                
                # find which phrase matches this key best
                key_emb = self.key_embeddings[idx]
                best_phrase = prompt  # default to full prompt
                best_phrase_sim = dense_sims[idx]
                
                # check each phrase
                for phrase in key_phrases:
                    phrase_emb = self.embed_service.embed_single(phrase)
                    phrase_sim = self.embed_service.cosine_similarity(phrase_emb, key_emb)
                    if phrase_sim > best_phrase_sim:
                        best_phrase = phrase
                        best_phrase_sim = phrase_sim
                
                mappings.append(KeyMapping(
                    user_phrase=best_phrase[:50],
                    mapped_to=self.keys[idx]['value'],
                    similarity=float(np.clip(normalized_score, 0.0, 1.0))
                ))
                
                if len(mappings) >= top_k:
                    break
            
            logger.success(f"Mapped {len(mappings)} keys successfully")
            return mappings
        
        except Exception as e:
            logger.error(f"Key mapping failed: {str(e)}")
            # return empty list if everything breaks
            return []
    
    def get_top_keys(self, prompt, top_k=5, min_similarity=None):
        """Get top keys with full metadata"""
        try:
            threshold = min_similarity if min_similarity is not None else self.threshold
            
            # get more than needed then filter
            mappings = self.map_keys(prompt, top_k=top_k * 2)
            
            filtered = [m for m in mappings if m.similarity >= threshold]
            
            # add full key details
            result = []
            for mapping in filtered[:top_k]:
                key_obj = next((k for k in self.keys if k['value'] == mapping.mapped_to), None)
                if key_obj:
                    result.append({
                        **key_obj,
                        'similarity': mapping.similarity,
                        'matched_phrase': mapping.user_phrase
                    })
            
            return result
        except Exception as e:
            logger.error(f"get_top_keys failed: {str(e)}")
            return []
