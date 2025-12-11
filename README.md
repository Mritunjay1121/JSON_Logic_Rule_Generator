# JSON Logic Rule Generator (with RAG & Embeddings)

An AI-powered FastAPI service that converts natural-language prompts into valid JSON Logic rules using embeddings, hybrid key mapping, and a lightweight RAG layer over domain policies. The API also returns a plain-English explanation, the keys used, and a confidence score for each generated rule.

## Features

- Natural language → JSON Logic rule generation
- Uses only predefined `SAMPLE_STORE_KEYS` in `{"var": "key"}` format
- Hybrid key mapping (embeddings + BM25 + Reciprocal Rank Fusion)
- Lightweight RAG over hard-coded policy documents (FAISS + CRAG-style refinement)
- Self-consistency: multiple rule variants with voting + mock-data validation
- Confidence score based on key similarity, policy relevance, and generation quality
- Utility endpoints to inspect keys and policies
- Ready for local dev, UV, or Docker usage

## Prerequisites

- Python 3.10 or higher
- pip or uv for dependency management
- An OpenAI API key (for GPT-4o-mini)

## Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/<your-username>/<your-repo>.git
cd jsonlogic-rag-api
```


### 2. Create a Python Environment

```
python -m venv venv
```


### 3. Activate the Environment and Install Requirements

#### On Windows

```
venv\Scripts\activate
```


#### On macOS / Linux

```
source venv/bin/activate
```


#### Install Requirements

```
pip install -r requirements.txt
```


### 4. Configure Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY="sk-your-key-here"
EMBED_MODEL="all-MiniLM-L6-v2"
RRF_K="60"
SIM_THRESHOLD="0.7"
```


### 5. Run the API with Uvicorn

```
python run.py
```


The API will start at:

- Swagger UI: http://127.0.0.1:8000/docs  
- Health check: http://127.0.0.1:8000/health  

### Or you can use Uvicorn directly

```
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Use below endpoint to get results. Here is the example input

```
http://127.0.0.1:8000/generate-rule
```

Example input

```
{
  "context_docs": [
    "Custom policy: Minimum age 25"
  ],
  "prompt": "Approve if bureau score > 700 and business vintage at least 3 years"
}
```



The response will include:

- `json_logic`: the generated JSON Logic rule
- `explanation`: short natural-language explanation
- `used_keys`: which fields were used
- `key_mappings`: how user phrases mapped to keys
- `confidence_score`: overall confidence (0–1)



### Test

To test it with some examples there are tests written in "tests" folder . To test just run below command

```
python tests\test_api.py
```