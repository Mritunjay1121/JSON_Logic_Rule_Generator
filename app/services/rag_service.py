import numpy as np
import faiss
import os
from openai import OpenAI
from loguru import logger
from app.services.embedding_service import EmbeddingService
from app.constants import POLICIES


class RAGService:
    """ Handles policy retrieval with FAISS 
    CRAG = corrective RAG, basically retries if results are bad """
    
    def __init__(self, embedding_service):
        try:
            self.embed_service = embedding_service
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            logger.info("Initializing RAG Service...")
            
            self.policies = POLICIES.copy()
            self.build_index()
            
            logger.success(f"RAG Service initialized with {len(self.policies)} policies")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {str(e)}")
            raise
    
    def build_index(self):
        """Build FAISS index from policy docs"""
        try:
            if not self.policies:
                logger.warning("No policies to index")
                self.index = None
                self.policy_embeddings = None
                return
            
            logger.info(f"Embedding {len(self.policies)} policy documents...")
            
            # embed all policies
            self.policy_embeddings = self.embed_service.embed_batch(self.policies)
            
            # FAISS index - using inner product since vectors are normalized
            dimension = self.embed_service.get_dimension()
            self.index = faiss.IndexFlatIP(dimension)
            
            self.index.add(self.policy_embeddings.astype('float32'))
            
            logger.debug(f"FAISS index built with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Index building failed: {str(e)}")
            self.index = None
            self.policy_embeddings = None
    
    def add_documents(self, new_docs):
        """Add new docs to index on the fly"""
        try:
            if not new_docs:
                return
            
            logger.info(f"Adding {len(new_docs)} temporary documents...")
            
            new_embeddings = self.embed_service.embed_batch(new_docs)
            self.index.add(new_embeddings.astype('float32'))
            self.policies.extend(new_docs)
            
            # stack new embeddings with old ones
            self.policy_embeddings = np.vstack([self.policy_embeddings, new_embeddings])
            
            logger.debug(f"Index now contains {self.index.ntotal} documents")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
    
    def retrieve(self, query, top_k=3):
        """Basic retrieval from FAISS"""
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Index is empty, returning no results")
                return []
            
            # embed and search
            query_emb = self.embed_service.embed_single(query).reshape(1, -1)
            scores, indices = self.index.search(query_emb.astype('float32'), top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.policies):
                    results.append({
                        'text': self.policies[idx],
                        'score': float(score),
                        'index': int(idx),
                        'rank': i + 1
                    })
            
            logger.debug(f"Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    def judge_relevance(self, query, documents):
        # use llm to score how relevant each doc is
        # helps filter out garbage results
        try:
            if not documents:
                return []
            
            doc_texts = "\n\n".join([
                f"DOCUMENT {i+1}:\n{doc['text']}" 
                for i, doc in enumerate(documents)
            ])
            
            judge_prompt = f"""You are an expert relevance evaluator for a loan application rule generation system.

QUERY: {query}

RETRIEVED DOCUMENTS:
{doc_texts}

Task: Rate the relevance of each document to the query on a scale of 0.0 to 1.0.
- 1.0 = Highly relevant, directly helps answer the query
- 0.5 = Somewhat relevant, provides context
- 0.0 = Not relevant at all

Respond ONLY with a JSON array of scores, one per document in order.
Example: [0.9, 0.6, 0.2]

Scores:"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a relevance scoring expert. Respond only with a JSON array of numbers."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            import json
            scores = json.loads(content)
            
            # clamp to 0-1 range
            scores = [max(0.0, min(1.0, float(s))) for s in scores]
            
            # pad if llm didnt return enough scores
            while len(scores) < len(documents):
                scores.append(0.5)
            
            logger.debug(f"LLM judge scores: {scores}")
            return scores[:len(documents)]
            
        except Exception as e:
            logger.error(f"LLM judge failed: {str(e)}")
            # fallback - just use retrieval scores
            return [doc['score'] / (doc['score'] + 1.0) for doc in documents]
    
    def refine_query(self, original_query, low_relevance_docs):
        # if results are bad, ask llm to rewrite the query
        # usually helps by adding more specific terms
        try:
            refine_prompt = f"""Original query: "{original_query}"

The retrieved documents were not very relevant. Suggest a better search query that focuses on key loan application terms like:
- Bureau score, credit score, CIBIL
- Business vintage, age
- Overdue amounts, DPD
- Income, FOIR
- GST, banking metrics

Respond with ONLY the improved query, no explanation.

Improved query:"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a query refinement expert for loan application rules."},
                    {"role": "user", "content": refine_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            refined = response.choices[0].message.content.strip().strip('"')
            logger.info(f"Refined query: {refined}")
            return refined if refined else original_query
            
        except Exception as e:
            logger.error(f"Query refinement failed: {str(e)}")
            return original_query
    
    def retrieve_with_crag(self, query, top_k=2, relevance_threshold=0.7):
        """
        CRAG = Corrective RAG
        retrieves docs, checks if theyre good, retries if not
        """
        try:
            logger.info(f"CRAG: Retrieving for query: '{query[:50]}...'")
            
            docs = self.retrieve(query, top_k=top_k)
            
            if not docs:
                logger.warning("No documents retrieved")
                return [], 0.0
            
            # judge how relevant results are
            relevance_scores = self.judge_relevance(query, docs)
            
            for doc, score in zip(docs, relevance_scores):
                doc['relevance'] = score
            
            avg_relevance = np.mean(relevance_scores)
            logger.debug(f"CRAG: Initial relevance: {avg_relevance:.3f}")
            
            # if relevance sucks, refine and try again
            if avg_relevance < relevance_threshold:
                logger.info("CRAG: Low relevance detected, refining query...")
                
                refined_query = self.refine_query(query, [d['text'] for d in docs])
                logger.debug(f"CRAG: Refined query: '{refined_query[:50]}...'")
                
                # try again with better query
                refined_docs = self.retrieve(refined_query, top_k=top_k)
                
                if refined_docs:
                    refined_relevance = self.judge_relevance(refined_query, refined_docs)
                    
                    for doc, score in zip(refined_docs, refined_relevance):
                        doc['relevance'] = score
                    
                    refined_avg = np.mean(refined_relevance)
                    logger.debug(f"CRAG: Refined relevance: {refined_avg:.3f}")
                    
                    # use refined results only if theyre better
                    if refined_avg > avg_relevance:
                        docs = refined_docs
                        avg_relevance = refined_avg
                        logger.info("CRAG: Using refined results")
                    else:
                        logger.info("CRAG: Keeping original results")
            
            # sort by relevance score
            docs.sort(key=lambda x: x['relevance'], reverse=True)
            
            return docs, avg_relevance
        
        except Exception as e:
            logger.error(f"CRAG failed: {str(e)}")
            return [], 0.0
    
    def format_context(self, documents, max_length=500):
        """Format docs into a string for llm context"""
        try:
            if not documents:
                return "No relevant policies found."
            
            context_parts = []
            for i, doc in enumerate(documents):
                text = doc['text'][:max_length]
                relevance = doc.get('relevance', doc.get('score', 0))
                context_parts.append(
                    f"Policy {i+1} (relevance: {relevance:.2f}):\n{text}"
                )
            
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Context formatting failed: {str(e)}")
            return "Error formatting policy context."
