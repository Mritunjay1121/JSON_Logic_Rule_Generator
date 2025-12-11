import json
import os
from openai import OpenAI
from json_logic import jsonLogic
from loguru import logger
from app.constants import MOCK_STORE_SAMPLES
from app.models import KeyMapping


class RuleGenerationService:
    """ Generates json logic rules using gpt-4o-mini
        Uses self-consistency voting to pick best rule from multiple attempts
    """
    
    def __init__(self):
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4o-mini"
            logger.success("RuleGenerationService initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RuleGenerationService: {str(e)}")
            raise
    
    def build_system_prompt(self, available_keys, policy_context):
        
        try:
            keys_str = json.dumps(available_keys, indent=2)
            
            system_prompt = f"""You are an expert JSON Logic rule generator for loan application systems.

AVAILABLE KEYS (use ONLY these in {{"var": "key"}}):
{keys_str}

POLICY CONTEXT:
{policy_context}

JSON LOGIC OPERATORS:
- Logical: "and", "or", "!", "if"
- Comparison: ">", "<", ">=", "<=", "==", "!="
- Arrays: "in", "some", "all"
- Math: "+", "-", "*", "/"

RULES:
1. Use ONLY the available keys listed above
2. All keys must be referenced using {{"var": "key.name"}}
3. Generate valid JSON Logic syntax
4. Be precise with thresholds from policies
5. Use "and" for multiple conditions, "or" for alternatives

OUTPUT FORMAT (must be valid JSON):
{{
  "json_logic": {{"and": [...]}},
  "explanation": "Brief 1-2 sentence explanation",
  "used_keys": ["key1", "key2"],
  "confidence": 0.0-1.0
}}

EXAMPLES:

User: "Approve if bureau score > 700"
Output:
{{
  "json_logic": {{">": [{{"var": "bureau.score"}}, 700]}},
  "explanation": "Approves applications where bureau score exceeds 700.",
  "used_keys": ["bureau.score"],
  "confidence": 0.95
}}

User: "Reject if wilful default OR suit filed"
Output:
{{
  "json_logic": {{"or": [
    {{"==": [{{"var": "bureau.wilful_default"}}, true]}},
    {{"==": [{{"var": "bureau.suit_filed"}}, true]}}
  ]}},
  "explanation": "Rejects applications with wilful default or suit filed status.",
  "used_keys": ["bureau.wilful_default", "bureau.suit_filed"],
  "confidence": 0.92
}}

User: "Approve if age between 25 and 60"
Output:
{{
  "json_logic": {{"and": [
    {{">=": [{{"var": "primary_applicant.age"}}, 25]}},
    {{"<=": [{{"var": "primary_applicant.age"}}, 60]}}
  ]}},
  "explanation": "Approves when primary applicant age is between 25 and 60 years inclusive.",
  "used_keys": ["primary_applicant.age"],
  "confidence": 0.90
}}"""

            return system_prompt
        except Exception as e:
            logger.error(f"Failed to build system prompt: {str(e)}")
            return ""
    
    def generate_single_rule(self, prompt, system_prompt, temperature=0.2):
        # calls llm once to get a rule
        # temperature controls randomness - lower = more deterministic
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=800,
                response_format={"type": "json_object"}  
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # check if all required fields are there
            if not all(k in result for k in ['json_logic', 'explanation', 'used_keys']):
                logger.warning("LLM response missing required fields")
                raise ValueError("Missing required fields in LLM response")
            
            # add default confidence if llm forgot to include it
            if 'confidence' not in result:
                result['confidence'] = 0.8
            
            return result
            
        except Exception as e:
            logger.error(f"Rule generation failed: {str(e)}")
            return None
    
    def validate_rule(self, rule, available_keys):
        # checks if rule only uses allowed keys
        # extracts all {"var": "..."} and validates them
        try:
            if not rule:
                return False, "Empty rule"
            
            # recursively find all var references
            def extract_vars(obj):
                vars_found = []
                if isinstance(obj, dict):
                    if "var" in obj:
                        vars_found.append(obj["var"])
                    for value in obj.values():
                        vars_found.extend(extract_vars(value))
                elif isinstance(obj, list):
                    for item in obj:
                        vars_found.extend(extract_vars(item))
                return vars_found
            
            used_vars = extract_vars(rule)
            
            # check all vars are in allowed list
            invalid_vars = [v for v in used_vars if v not in available_keys]
            if invalid_vars:
                logger.warning(f"Rule uses invalid keys: {invalid_vars}")
                return False, f"Invalid keys used: {invalid_vars}"
            
            return True, ""
        
        except Exception as e:
            logger.error(f"Rule validation failed: {str(e)}")
            return False, str(e)
    
    def test_rule_on_mocks(self, rule, num_samples=5):
        # runs the rule against mock data to see if it breaks
        # doesn't check correctness, just that it executes
        try:
            if not rule:
                return 0.0
            
            successes = 0
            samples = MOCK_STORE_SAMPLES[:num_samples]
            
            for sample in samples:
                try:
                    # apply json logic rule
                    result = jsonLogic(rule, sample)
                    successes += 1
                except Exception as e:
                    # rule broke on this sample
                    logger.debug(f"Rule test failed on sample: {str(e)}")
                    continue
            
            success_rate = successes / len(samples) if samples else 0.0
            return success_rate
        
        except Exception as e:
            logger.error(f"Mock testing failed: {str(e)}")
            return 0.0
    
    def self_consistency_vote(self, variants):
        # picks the best rule from multiple variants
        # scores based on confidence, validation rate, and simplicity
        try:
            if not variants:
                return None
            
            if len(variants) == 1:
                return variants[0]
            
            scored_variants = []
            for variant in variants:
                score = 0.0
                
                # llm's own confidence
                score += variant.get('confidence', 0.5) * 0.4
                
                # how well it ran on mock data
                validation_rate = variant.get('validation_rate', 0.0)
                score += validation_rate * 0.4
                
                # prefer simpler rules (less json length)
                rule_str = json.dumps(variant['json_logic'])
                complexity_penalty = min(len(rule_str) / 500, 0.2)
                score += (0.2 - complexity_penalty)
                
                scored_variants.append((score, variant))
            
            # sort by score descending
            scored_variants.sort(key=lambda x: x[0], reverse=True)
            
            scores_str = [f'{s:.3f}' for s, _ in scored_variants]
            logger.debug(f"Self-consistency scores: {scores_str}")
            
            return scored_variants[0][1]  # return best one
        
        except Exception as e:
            logger.error(f"Self-consistency voting failed: {str(e)}")
            return variants[0] if variants else None
    
    def generate_rule(self, prompt, key_mappings, policy_context, num_variants=3):
        """
        Main method - generates rule with self-consistency
        tries multiple times with different temperatures and picks best
        """
        try:
            logger.info(f"Generating {num_variants} rule variants...")
            
            # get list of allowed keys
            available_keys = [m.mapped_to for m in key_mappings]
            
            # build the big system prompt
            system_prompt = self.build_system_prompt(available_keys, policy_context)
            
            # generate multiple variants
            variants = []
            temperatures = [0.1, 0.3, 0.5][:num_variants]
            
            for i, temp in enumerate(temperatures):
                logger.debug(f"Generating variant {i+1} with temp={temp}...")
                
                result = self.generate_single_rule(prompt, system_prompt, temperature=temp)
                
                if result:
                    # validate it uses correct keys
                    is_valid, error_msg = self.validate_rule(
                        result['json_logic'], 
                        available_keys
                    )
                    
                    if not is_valid:
                        logger.warning(f"Variant {i+1} validation failed: {error_msg}")
                        continue
                    
                    # test on mock data
                    validation_rate = self.test_rule_on_mocks(result['json_logic'])
                    result['validation_rate'] = validation_rate
                    
                    logger.info(f"Variant {i+1}: conf={result['confidence']:.3f}, val={validation_rate:.3f}")
                    
                    variants.append(result)
            
            if not variants:
                logger.error("Failed to generate any valid rule variants")
                raise ValueError("Failed to generate any valid rule variants")
            
            # vote for best rule
            best_rule = self.self_consistency_vote(variants)
            
            logger.success(f"Selected best rule")
            
            return best_rule
        
        except Exception as e:
            logger.error(f"Rule generation failed: {str(e)}")
            raise
    
    def calculate_confidence_score(self, rule_result, key_mappings, policy_relevance):
        """Calculate overall confidence from multiple factors"""
        try:
            # how well keys matched (40%)
            avg_key_sim = sum(m.similarity for m in key_mappings) / len(key_mappings) if key_mappings else 0.0
            
            # how relevant policies were (30%)
            policy_score = policy_relevance
            
            # llm confidence + validation rate (30%)
            llm_confidence = rule_result.get('confidence', 0.8)
            validation_rate = rule_result.get('validation_rate', 0.8)
            generation_score = (llm_confidence + validation_rate) / 2
            
            # weighted average
            confidence = (
                avg_key_sim * 0.4 +
                policy_score * 0.3 +
                generation_score * 0.3
            )
            
            # clamp to 0-1
            return float(min(max(confidence, 0.0), 1.0))
        
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.5  # default fallback
