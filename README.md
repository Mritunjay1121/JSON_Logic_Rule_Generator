# JSON Logic Rule Generator (with RAG & Embeddings)

An AI-powered FastAPI service that converts natural-language prompts into valid JSON Logic rules using embeddings, hybrid key mapping, and a lightweight RAG layer over domain policies. The API also returns a plain-English explanation, the keys used, and a confidence score for each generated rule.

## Features

- Natural language → JSON Logic rule generation
- Uses only predefined `SAMPLE_STORE_KEYS` in `{"var": "key"}` format
- Hybrid key mapping (embeddings + BM25 + Reciprocal Rank Fusion)
- Lightweight RAG over hard-coded policy documents (FAISS + Corrective RAG-style refinement)
- Self-consistency: multiple rule variants with voting + mock-data validation
- Confidence score based on key similarity, policy relevance, and generation quality
- Utility endpoints to inspect keys and policies
- Ready for local dev, UV, or Docker usage

## Prerequisites

- Python 3.10 or higher
- pip for dependency management
- An OpenAI API key (for GPT-4o-mini)


## 🧠 Architecture & Flow

### High-Level Architecture


### 1. Prompt In → Key Mapping

- Take the user’s natural-language prompt.
- Break it into meaningful phrases like:
  - “bureau score”
  - “business vintage”
  - “age between 25 and 60”
- For each phrase:
  - Compute a semantic similarity score using embeddings.
  - Compute a keyword score using BM25.
- Combine both scores and map phrases to allowed keys, for example:
  - “bureau score” → `bureau.score`
  - “business vintage” → `business.vintage_in_years`
  - “age” → `primary_applicant.age`

---

### 2. Policy Retrieval (RAG Layer)

- Embed the full prompt and search against a small set of hard‑coded policy documents.
- Pick the top few policies that are most relevant to the prompt.
- Join them into a short context block that captures things like:
  - Minimum bureau scores
  - Allowed age ranges
  - Required vintage, income, etc.
- This context guides the model to choose realistic thresholds.

---

### 3. Rule Generation (LLM + Self‑Consistency)

- Send three things to the model:
  - The original prompt
  - The mapped keys
  - The policy context
- Ask it to return a JSON object containing:
  - `json_logic`
  - `explanation`
  - `used_keys`
  - `confidence`
- Generate multiple variants with different temperatures.
- For each variant:
  - Check that it is valid JSON.
  - Check that it only uses keys from `SAMPLE_STORE_KEYS`.
  - Run it on some mock data to ensure it executes without errors.
- Keep the variant that scores best across these checks.

---

### 4. Confidence Scoring

- Compute three partial scores:
  - **Key similarity**: how well phrases mapped to keys.
  - **Policy relevance**: how relevant the retrieved policies were.
  - **Generation quality**: model‑reported confidence + mock validation rate.
- Combine them into a single `confidence_score` between 0 and 1 using fixed weights.

---

### 5. Response Out

The API returns a single, structured result:

- `json_logic` – the final selected rule.
- `explanation` – a short, human‑readable summary of the rule.
- `used_keys` – list of all keys actually used in the rule.
- `key_mappings` – how phrases from the prompt mapped to keys, with similarity scores.
- `confidence_score` – overall trust score for this particular rule.


## Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/Mritunjay1121/JSON_Logic_Rule_Generator.git
cd JSON_Logic_Rule_Generator
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

### Use below Endpoints to get results and see configurations. 


#### ROOT /

**Response:**

```
{
  "status": "online",
  "service": "JSON Logic Rule Generator API",
  "version": "1.0.0",
  "endpoints": {
    "generate_rule": "/generate-rule",
    "docs": "/docs",
    "health": "/health"
  }
}
```

#### POST /generate-rule

```
http://127.0.0.1:8000/generate-rule
```

Example input

**Request Body:**

```
{
  "context_docs": [
    "Custom policy: Minimum age 25"
  ],
  "prompt": "Approve if bureau score > 700 and business vintage at least 3 years"
}
```


OR Use CURL 


```
curl -X 'POST' \
  'https://datasciencesage-json-logic-rule-generator.hf.space/generate-rule' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "context_docs": [
    "Custom policy: Minimum age 25"
  ],
  "prompt": "Approve if bureau score > 700 and business vintage at least 3 years"
}'
```

Output :

```
{
  "json_logic": {
    "and": [
      {
        ">": [
          {
            "var": "bureau.score"
          },
          700
        ]
      },
      {
        ">=": [
          {
            "var": "business.vintage_in_years"
          },
          3
        ]
      }
    ]
  },
  "explanation": "Approves applications where the bureau score exceeds 700 and the business has been established for at least 3 years.",
  "used_keys": [
    "bureau.score",
    "business.vintage_in_years"
  ],
  "key_mappings": [
    {
      "user_phrase": "business vintage",
      "mapped_to": "business.vintage_in_years",
      "similarity": 0.9919354838709679
    },
    {
      "user_phrase": "bureau score",
      "mapped_to": "bureau.score",
      "similarity": 0.9919354838709679
    },
    {
      "user_phrase": "Approve if bureau score > 700 and business vintage",
      "mapped_to": "business.commercial_cibil_score",
      "similarity": 0.9533577533577533
    },
    {
      "user_phrase": "Approve if bureau score > 700 and business vintage",
      "mapped_to": "itr.years_filed",
      "similarity": 0.910650623885918
    },
    {
      "user_phrase": "approve if bureau",
      "mapped_to": "bureau.enquiries",
      "similarity": 0.9104477611940298
    }
  ],
  "confidence_score": 0.823166168494371
}
```


The response will include:

- `json_logic`: the generated JSON Logic rule
- `explanation`: short natural-language explanation
- `used_keys`: which fields were used
- `key_mappings`: how user phrases mapped to keys
- `confidence_score`: overall confidence (0–1)

### Other Endpoints


#### GET /health

Service health check.

**Response:**

```
{
"status": "healthy",
"services": {
"embedding": true,
"key_mapper": true,
"rag": true,
"rule_generation": true
},
"models": {
"embedding_model": "all-MiniLM-L6-v2",
"llm_model": "gpt-4o-mini"
}
}
```

#### GET /policies

View policy documents used by RAG system.

**Response:**

```
{
"total_policies": 10,
"policies": [
{"id": 1, "text": "Applicants must be between 21 and 65 years old"},
{"id": 2, "text": "Minimum bureau score of 650 required for approval"}
]
}
```


#### GET /keys

List all available predefined keys grouped by category.

**Response:**

```
{
"total_keys": 25,
"groups": ["Applicant", "Business", "Financial", "Credit"],
"keys_by_group": {
"Applicant": [
{"value": "applicant_age", "label": "Applicant Age"},
{"value": "applicant_gender", "label": "Applicant Gender"}
],
"Business": [
{"value": "business_vintage_months", "label": "Business Vintage (months)"},
{"value": "business_type", "label": "Business Type"}
]
}
}
```


### Test

To test it with some examples there are tests written in "tests" folder . To test just run below command

```
python tests\test_api.py
```


### DEPLOYED IN HUGGINGFACE SPACES . PLEASE DO HAVE A TEST :

#### HUGGINGFACE REPO :

```
https://huggingface.co/spaces/datasciencesage/JSON_Logic_Rule_Generator
```

#### USE BELOW FOR TESTING :

Deployed on HUGGINGFACE SPACES : [Deployed](https://datasciencesage-json-logic-rule-generator.hf.space/)
