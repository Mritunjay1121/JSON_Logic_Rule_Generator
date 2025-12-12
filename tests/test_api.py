import pytest
from fastapi.testclient import TestClient
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

# Create client with context manager to trigger lifespan
@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan events"""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test root endpoint returns service info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data


def test_list_keys(client):
    """Test listing available keys"""
    response = client.get("/keys")
    assert response.status_code == 200
    data = response.json()
    assert data["total_keys"] == 37
    assert "keys_by_group" in data


def test_list_policies(client):
    """Test listing policies"""
    response = client.get("/policies")
    assert response.status_code == 200
    data = response.json()
    assert data["total_policies"] >= 10
    assert isinstance(data["policies"], list)




def test_generate_rule_example1(client):
    """Test Example 1: Bureau score and business vintage"""
    payload = {
        "prompt": "Approve if bureau score > 700 and business vintage at least 3 years and applicant age between 25 and 60"
    }
    
    response = client.post("/generate-rule", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Print required outputs
    print("\n" + "="*70)
    print("EXAMPLE 1: Bureau Score + Vintage + Age")
    print("="*70)
    print(f"\n📝 PROMPT:")
    print(f"   {payload['prompt']}")
    print(f"\n⚙️  JSON LOGIC:")
    print(json.dumps(data['json_logic'], indent=2))
    print(f"\n💬 EXPLANATION:")
    print(f"   {data['explanation']}")
    print(f"\n🔑 KEY MAPPINGS:")
    for mapping in data['key_mappings']:
        print(f"   - '{mapping['user_phrase']}' → {mapping['mapped_to']} (similarity: {mapping['similarity']:.3f})")
    print(f"\n📊 USED KEYS: {', '.join(data['used_keys'])}")
    print(f"\n✅ CONFIDENCE SCORE: {data['confidence_score']:.3f}")
    print("="*70)
    
    # Check structure
    assert "json_logic" in data
    assert "explanation" in data
    assert "used_keys" in data
    assert "key_mappings" in data
    assert "confidence_score" in data
    
    # Check keys used
    assert "bureau.score" in data["used_keys"]
    assert "business.vintage_in_years" in data["used_keys"]
    
    # Check confidence
    assert 0.0 <= data["confidence_score"] <= 1.0
    
    # Check key mappings
    assert len(data["key_mappings"]) > 0
    for mapping in data["key_mappings"]:
        assert "user_phrase" in mapping
        assert "mapped_to" in mapping
        assert "similarity" in mapping


def test_generate_rule_example2(client):
    """Test Example 2: High risk flags (OR conditions)"""
    payload = {
        "prompt": "Flag as high risk if wilful default is true OR overdue amount > 50000 OR bureau dpd >= 90"
    }
    
    response = client.post("/generate-rule", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Print required outputs
    print("\n" + "="*70)
    print("EXAMPLE 2: High Risk Detection (OR Logic)")
    print("="*70)
    print(f"\n📝 PROMPT:")
    print(f"   {payload['prompt']}")
    print(f"\n⚙️  JSON LOGIC:")
    print(json.dumps(data['json_logic'], indent=2))
    print(f"\n💬 EXPLANATION:")
    print(f"   {data['explanation']}")
    print(f"\n🔑 KEY MAPPINGS:")
    for mapping in data['key_mappings']:
        print(f"   - '{mapping['user_phrase']}' → {mapping['mapped_to']} (similarity: {mapping['similarity']:.3f})")
    print(f"\n📊 USED KEYS: {', '.join(data['used_keys'])}")
    print(f"\n✅ CONFIDENCE SCORE: {data['confidence_score']:.3f}")
    print("="*70)
    
    # Check OR logic is used
    rule = data["json_logic"]
    assert "or" in json.dumps(rule).lower()
    
    # Check keys
    assert "bureau.wilful_default" in data["used_keys"]
    assert "bureau.overdue_amount" in data["used_keys"]


def test_generate_rule_example3(client):
    """Test Example 3: Tag-based or income rule"""
    payload = {
        "prompt": "Prefer applicants with tag veteran OR with monthly_income > 100000"
    }
    
    response = client.post("/generate-rule", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Print required outputs
    print("\n" + "="*70)
    print("EXAMPLE 3: Tag-Based Preference")
    print("="*70)
    print(f"\n📝 PROMPT:")
    print(f"   {payload['prompt']}")
    print(f"\n⚙️  JSON LOGIC:")
    print(json.dumps(data['json_logic'], indent=2))
    print(f"\n💬 EXPLANATION:")
    print(f"   {data['explanation']}")
    print(f"\n🔑 KEY MAPPINGS:")
    for mapping in data['key_mappings']:
        print(f"   - '{mapping['user_phrase']}' → {mapping['mapped_to']} (similarity: {mapping['similarity']:.3f})")
    print(f"\n📊 USED KEYS: {', '.join(data['used_keys'])}")
    print(f"\n✅ CONFIDENCE SCORE: {data['confidence_score']:.3f}")
    print("="*70)
    
    # Check keys
    assert "primary_applicant.tags" in data["used_keys"]
    assert "primary_applicant.monthly_income" in data["used_keys"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
