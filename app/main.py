from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Add parent directory to Python path (universal fix)
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Now use absolute imports
from app.models import GenerateRuleRequest, GenerateRuleResponse, ErrorResponse
from app.services.embedding_service import EmbeddingService
from app.services.key_mapper import KeyMapper
from app.services.rag_service import RAGService
from app.services.rule_service import RuleGenerationService

# Load environment variables
load_dotenv()

# Global service instances
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown handler"""
    # Startup - load all services
    logger.info("="*50)
    logger.info("Initializing JSON Logic Rule Generator API")
    logger.info("="*50)
    
    try:
        # Initialize embedding service
        logger.info("1. Initializing Embedding Service...")
        embedding_service = EmbeddingService()
        services['embedding'] = embedding_service
        
        # Initialize key mapper
        logger.info("2. Initializing Key Mapper...")
        key_mapper = KeyMapper(embedding_service)
        services['key_mapper'] = key_mapper
        
        # Initialize RAG service
        logger.info("3. Initializing RAG Service...")
        rag_service = RAGService(embedding_service)
        services['rag'] = rag_service
        
        # Initialize rule service
        logger.info("4. Initializing Rule Generation Service...")
        rule_service = RuleGenerationService()
        services['rule'] = rule_service
        
        logger.success("="*50)
        logger.success("All services initialized successfully!")
        logger.success("API ready to accept requests")
        logger.success("="*50)
        
    except Exception as e:
        logger.error(f"FATAL ERROR during initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    services.clear()


# Create FastAPI app
app = FastAPI(
    title="JSON Logic Rule Generator API",
    description="AI-powered API for generating JSON Logic rules from natural language with RAG & embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - allow everything for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_services():
    """DI for services"""
    return services


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "JSON Logic Rule Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate_rule": "/generate-rule",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check(svc: Dict = Depends(get_services)):
    """Health check - shows which services are loaded"""
    return {
        "status": "healthy",
        "services": {
            "embedding": "embedding" in svc,
            "key_mapper": "key_mapper" in svc,
            "rag": "rag" in svc,
            "rule_generation": "rule" in svc
        },
        "models": {
            "embedding_model": os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
            "llm_model": "gpt-4o-mini"
        }
    }


@app.post(
    "/generate-rule",
    response_model=GenerateRuleResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Rule Generation"]
)
async def generate_rule(
    request: GenerateRuleRequest,
    svc: Dict = Depends(get_services)
) -> GenerateRuleResponse:
    """
    Generate JSON Logic rule from natural language
    
    Process:
    1. Maps user phrases to store keys (hybrid: embeddings + BM25 + RRF)
    2. Retrieves relevant policies using CRAG
    3. Generates JSON Logic with self-consistency voting
    4. Validates on mock data
    
    Returns valid JSON Logic + explanation + confidence score
    """
    start_time = time.time()
    
    try:
        logger.info("="*60)
        logger.info(f"NEW REQUEST: {request.prompt[:80]}...")
        logger.info("="*60)
        
        # grab services
        key_mapper = svc['key_mapper']
        rag_service = svc['rag']
        rule_service = svc['rule']
        
        # Step 1: map keys
        logger.info("[1/4] Mapping user phrases to keys...")
        key_mappings = key_mapper.map_keys(request.prompt, top_k=5)
        
        if not key_mappings:
            # nothing found - suggest closest matches
            all_mappings = key_mapper.map_keys(request.prompt, top_k=3)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No matching keys found",
                    "detail": "Prompt contains terms that couldn't be mapped to available keys",
                    "suggestions": [
                        {
                            "key": m.mapped_to,
                            "similarity": m.similarity,
                            "phrase": m.user_phrase
                        }
                        for m in all_mappings
                    ]
                }
            )
        
        logger.debug(f"Found {len(key_mappings)} key mappings")
        for m in key_mappings[:3]:
            logger.debug(f"  - {m.mapped_to}: {m.similarity:.3f}")
        
        # Step 2: add extra context if provided
        if request.context_docs:
            logger.info(f"[2/4] Adding {len(request.context_docs)} context documents...")
            rag_service.add_documents(request.context_docs)
        
        # Step 3: get relevant policies
        logger.info("[3/4] Retrieving relevant policies (CRAG)...")
        policy_docs, policy_relevance = rag_service.retrieve_with_crag(
            request.prompt,
            top_k=2
        )
        policy_context = rag_service.format_context(policy_docs)
        logger.debug(f"Policy relevance: {policy_relevance:.3f}")
        
        # Step 4: generate the rule
        logger.info("[4/4] Generating JSON Logic rule...")
        rule_result = rule_service.generate_rule(
            prompt=request.prompt,
            key_mappings=key_mappings,
            policy_context=policy_context,
            num_variants=3
        )
        
        # calculate final confidence
        confidence_score = rule_service.calculate_confidence_score(
            rule_result,
            key_mappings,
            policy_relevance
        )
        
        # build response
        response = GenerateRuleResponse(
            json_logic=rule_result['json_logic'],
            explanation=rule_result['explanation'],
            used_keys=rule_result['used_keys'],
            key_mappings=key_mappings,
            confidence_score=confidence_score
        )
        
        elapsed = time.time() - start_time
        logger.success("="*60)
        logger.success(f"SUCCESS - Generated rule in {elapsed:.2f}s")
        logger.success(f"Confidence: {confidence_score:.3f}")
        logger.success("="*60)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "detail": str(e)
            }
        )


@app.get("/keys", tags=["Utilities"])
async def list_available_keys():
    """List all available keys grouped by category"""
    from app.constants import SAMPLE_STORE_KEYS
    
    # group by category
    grouped = {}
    for key in SAMPLE_STORE_KEYS:
        group = key['group']
        if group not in grouped:
            grouped[group] = []
        grouped[group].append({
            'value': key['value'],
            'label': key['label']
        })
    
    return {
        "total_keys": len(SAMPLE_STORE_KEYS),
        "groups": list(grouped.keys()),
        "keys_by_group": grouped
    }


@app.get("/policies", tags=["Utilities"])
async def list_policies():
    """List all policy documents used by RAG"""
    from app.constants import POLICIES
    
    return {
        "total_policies": len(POLICIES),
        "policies": [
            {"id": i+1, "text": policy}
            for i, policy in enumerate(POLICIES)
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
