"""LLM client for parsing quiz instructions and solving problems using OpenRouter."""
import httpx
import asyncio
from typing import Optional, Dict, Any, List
import logging
import json
import re
import base64
from bs4 import BeautifulSoup
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, MAX_BUDGET_USD

logger = logging.getLogger(__name__)

# Cost tracking
_total_cost = 0.0
_selected_model = None
_model_pricing_cache = {}  # Cache model pricing for accurate cost estimation

# Initialize OpenRouter client
if OPENROUTER_API_KEY:
    client = httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "LLM Analysis Quiz Solver"
        },
        timeout=60.0
    )
else:
    client = None
    logger.warning("OpenRouter API key not set. LLM features will not work.")


async def fetch_available_models() -> List[Dict[str, Any]]:
    """
    Fetch available models from OpenRouter with pricing information.
    
    Returns:
        List of model dictionaries with pricing and capabilities
    """
    if not client:
        return []
    
    try:
        response = await client.get("/models")
        response.raise_for_status()
        models_data = response.json()
        return models_data.get("data", [])
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return []


async def select_best_model(task_type: str = "general") -> str:
    """
    Select the best model based on cost and capability.
    Prioritizes cost-effective models that can handle the task.
    
    Args:
        task_type: Type of task (parsing, analysis, transcription, vision, etc.)
    
    Returns:
        Model identifier string
    """
    global _selected_model
    
    # For transcription and vision, don't cache - need specialized models
    if _selected_model and task_type not in ("transcription", "vision"):
        return _selected_model
    
    if LLM_MODEL:
        if task_type not in ("transcription", "vision"):
            _selected_model = LLM_MODEL
        logger.info(f"Using configured model: {LLM_MODEL}")
        return LLM_MODEL
    
    models = await fetch_available_models()
    
    if not models:
        # Fallback based on task type
        if task_type == "transcription":
            logger.warning("Could not fetch models, using Whisper as default")
            return "openai/whisper-large-v3"
        elif task_type == "vision":
            logger.warning("Could not fetch models, using GPT-4 Vision as default")
            return "openai/gpt-4o"  # GPT-4o has vision
        else:
            logger.warning("Could not fetch models, using GPT-4 as default")
            return "openai/gpt-4-turbo"
    
    # For transcription tasks, prioritize Whisper models
    if task_type == "transcription":
        transcription_models = [
            "openai/whisper-large-v3",
            "openai/whisper-large-v2",
            "openai/whisper-large",
            "openai/whisper-medium",
        ]
        for preferred in transcription_models:
            for model in models:
                model_id = model.get("id", "")
                if model_id == preferred or preferred in model_id.lower():
                    logger.info(f"Selected transcription model: {model_id}")
                    return model_id
        # Fallback to any whisper model
        for model in models:
            model_id = model.get("id", "")
            if "whisper" in model_id.lower():
                logger.info(f"Selected transcription model: {model_id}")
                return model_id
    
    # For vision tasks, prioritize vision-capable models
    if task_type == "vision":
        vision_models = [
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            "openai/gpt-4-vision-preview",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "google/gemini-pro-vision",
            "google/gemini-pro-1.5",
        ]
        for preferred in vision_models:
            for model in models:
                model_id = model.get("id", "")
                if model_id == preferred or preferred in model_id.lower():
                    pricing = model.get("pricing", {})
                    if pricing:
                        logger.info(f"Selected vision model: {model_id}")
                        return model_id
    
    # Prioritize high-quality models: GPT-4, Claude 3.5, etc.
    # Quality is more important than cost for reliability
    
    preferred_models = [
        "openai/gpt-4-turbo",
        "openai/gpt-4",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "google/gemini-pro-1.5",
        "google/gemini-pro",
        "openai/gpt-3.5-turbo"
    ]
    
    suitable_models = []
    
    # First, try to find preferred models
    for preferred in preferred_models:
        for model in models:
            model_id = model.get("id", "")
            if model_id == preferred or preferred in model_id.lower():
                pricing = model.get("pricing", {})
                if pricing:
                    try:
                        prompt_price = float(pricing.get("prompt", 0)) if pricing.get("prompt") else 0.0
                        completion_price = float(pricing.get("completion", 0)) if pricing.get("completion") else 0.0
                        total_price_per_1k = (prompt_price + completion_price) / 1000
                        context_length = model.get("context_length", 0)
                        
                        suitable_models.append({
                            "id": model_id,
                            "prompt_price": prompt_price,
                            "completion_price": completion_price,
                            "total_price_per_1k": total_price_per_1k,
                            "context_length": context_length,
                            "name": model.get("name", model_id),
                            "priority": preferred_models.index(preferred) if preferred in preferred_models else 999
                        })
                        break
                    except (ValueError, TypeError):
                        continue
    
    # If no preferred models found, fallback to any capable model
    if not suitable_models:
        logger.warning("No preferred models found, searching for any capable model")
        for model in models:
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})
            if pricing:
                try:
                    prompt_price = float(pricing.get("prompt", 0)) if pricing.get("prompt") else 0.0
                    completion_price = float(pricing.get("completion", 0)) if pricing.get("completion") else 0.0
                    total_price_per_1k = (prompt_price + completion_price) / 1000
                    context_length = model.get("context_length", 0)
                    
                    # Accept models under reasonable price (up to $0.05 per 1K for quality)
                    if total_price_per_1k < 0.05 and context_length >= 8000:
                        suitable_models.append({
                            "id": model_id,
                            "prompt_price": prompt_price,
                            "completion_price": completion_price,
                            "total_price_per_1k": total_price_per_1k,
                            "context_length": context_length,
                            "name": model.get("name", model_id),
                            "priority": 999
                        })
                except (ValueError, TypeError):
                    continue
    
    if not suitable_models:
        # Final fallback to GPT-4
        logger.warning("No models found, using GPT-4 as fallback")
        _selected_model = "openai/gpt-4-turbo"
        return _selected_model
    
    # Sort by priority first (lower is better), then by context length
    suitable_models.sort(key=lambda x: (x.get("priority", 999), -x["context_length"]))
    
    selected = suitable_models[0]
    _selected_model = selected["id"]
    
    # Cache pricing for this model
    _model_pricing_cache[_selected_model] = {
        "prompt_per_1m": selected["prompt_price"],
        "completion_per_1m": selected["completion_price"]
    }
    
    logger.info(f"Selected model: {_selected_model} (${selected['total_price_per_1k']:.4f} per 1K tokens, {selected['context_length']:,} context)")
    return _selected_model


def estimate_cost(prompt_tokens: int, completion_tokens: int, model_id: str) -> float:
    """
    Estimate cost for API call using cached pricing or fallback estimates.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model_id: Model identifier
    
    Returns:
        Estimated cost in USD
    """
    # Try to use cached pricing first
    if model_id in _model_pricing_cache:
        pricing = _model_pricing_cache[model_id]
        prompt_price_per_1m = pricing["prompt_per_1m"]
        completion_price_per_1m = pricing["completion_per_1m"]
    else:
        # Fallback to conservative estimates based on model name
        if "gpt-3.5" in model_id.lower():
            prompt_price_per_1m = 0.5
            completion_price_per_1m = 1.5
        elif "gpt-4" in model_id.lower():
            prompt_price_per_1m = 10.0
            completion_price_per_1m = 30.0
        elif "claude" in model_id.lower() or "haiku" in model_id.lower():
            prompt_price_per_1m = 0.25
            completion_price_per_1m = 1.25
        elif "gemini" in model_id.lower():
            prompt_price_per_1m = 0.5
            completion_price_per_1m = 1.5
        elif "llama" in model_id.lower():
            prompt_price_per_1m = 0.1
            completion_price_per_1m = 0.1
        else:
            # Default to conservative cheap estimate
            prompt_price_per_1m = 0.5
            completion_price_per_1m = 1.5
    
    cost = (prompt_tokens / 1_000_000 * prompt_price_per_1m) + \
           (completion_tokens / 1_000_000 * completion_price_per_1m)
    
    return cost


def check_budget(cost: float) -> bool:
    """
    Check if we can afford this cost.
    
    Args:
        cost: Estimated cost in USD
    
    Returns:
        True if within budget, False otherwise
    """
    global _total_cost
    if _total_cost + cost > MAX_BUDGET_USD:
        logger.warning(f"Budget exceeded: ${_total_cost + cost:.4f} > ${MAX_BUDGET_USD}")
        return False
    return True


def add_cost(cost: float):
    """Add cost to total."""
    global _total_cost
    _total_cost += cost
    logger.info(f"Total cost so far: ${_total_cost:.4f} / ${MAX_BUDGET_USD}")


async def parse_quiz_instructions(html_content: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Parse quiz instructions from HTML content using LLM with retry logic.
    
    Args:
        html_content: Rendered HTML content from quiz page
        max_retries: Maximum number of retry attempts
    
    Returns:
        Dictionary with parsed instructions including:
        - question: The question text
        - submit_url: URL to submit answer to
        - data_sources: List of data sources mentioned
        - task_type: Type of task (scraping, analysis, visualization, etc.)
    """
    if not client:
        raise ValueError("OpenRouter API key not configured")
    
    # Select appropriate model
    model_id = await select_best_model("parsing")
    
    # Limit content to save tokens
    limited_content = html_content[:6000]
    
    # Extract base64 content and visible text first
    soup = BeautifulSoup(html_content, 'html.parser')
    decoded_text = ""
    for script in soup.find_all('script'):
        script_text = script.string or ""
        atob_match = re.search(r'atob\(`([^`]+)`\)', script_text)
        if atob_match:
            try:
                decoded = base64.b64decode(atob_match.group(1)).decode('utf-8')
                decoded_text += decoded + "\n"
            except:
                pass
    
    result_divs = soup.find_all(id=re.compile(r'result', re.I))
    for div in result_divs:
        decoded_text += div.get_text() + "\n"
    
    # Use decoded text if available, otherwise use HTML
    content_to_parse = decoded_text if decoded_text else limited_content
    
    prompt = f"""Parse the following quiz page content and extract the key information.
Return ONLY a valid JSON object with these exact fields:
{{
  "question": "The main question or task description",
  "submit_url": "The URL where answers should be submitted",
  "data_sources": ["list of URLs or files mentioned"],
  "task_type": "one of: scraping, api_call, data_cleansing, data_analysis, visualization, transcription, vision",
  "expected_answer_type": "one of: boolean, number, string, base64, json",
  "instructions": "Any specific instructions"
}}

Content:
{content_to_parse[:5000]}

Return ONLY valid JSON, no markdown, no explanations, no code blocks."""

    for attempt in range(max_retries):
        try:
            # Estimate cost before calling
            estimated_tokens = len(prompt.split()) * 1.3
            estimated_cost = estimate_cost(int(estimated_tokens), 1000, model_id)
            
            if not check_budget(estimated_cost):
                raise ValueError(f"Estimated cost ${estimated_cost:.4f} exceeds budget")
            
            response = await client.post(
                "/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "You are a JSON parser. Return ONLY valid JSON. No markdown, no code blocks, no explanations. Just the JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1500
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Track actual cost
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            actual_cost = estimate_cost(prompt_tokens, completion_tokens, model_id)
            add_cost(actual_cost)
            
            # Get the raw content from LLM
            if "choices" not in result or len(result["choices"]) == 0:
                logger.error(f"No choices in LLM response: {result}")
                raise ValueError("LLM returned no choices")
            
            raw_content = result["choices"][0].get("message", {}).get("content", "")
            if not raw_content:
                logger.error(f"Empty content in LLM response: {result}")
                raise ValueError("LLM returned empty content")
            
            
            content = raw_content.strip()
            
            if not content or len(content) < 5:
                logger.error(f"Empty or too short content from LLM: {content}")
                raise ValueError("LLM returned empty content")
            
            # Remove markdown code blocks
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()
            
            if not content or len(content) < 5:
                logger.error("Content became empty after markdown removal")
                raise ValueError("Content empty after processing")
            
            # Extract JSON object - try multiple patterns with better matching
            json_match = None
            patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
                r'\{.*?\}',  # Simple object (non-greedy)
                r'\{[^}]*\}',  # Any object
            ]
            
            extracted_content = content
            for pattern in patterns:
                matches = list(re.finditer(pattern, content, re.DOTALL))
                # Try the longest match first
                if matches:
                    matches.sort(key=lambda m: len(m.group(0)), reverse=True)
                    for match in matches:
                        extracted = match.group(0).strip()
                        if len(extracted) > 20:  # Make sure it's substantial
                            extracted_content = extracted
                            json_match = match
                            break
                    if json_match:
                        break
            
            # If no JSON found, use original content
            if not json_match or len(extracted_content.strip()) < 10:
                logger.warning("No JSON object found, using full content")
                extracted_content = content
            
            content = extracted_content
            
            # Final safety check
            if not content or len(content.strip()) < 5:
                logger.error("Content is empty after extraction")
                raise ValueError("No valid content to parse")
            
            # Try to parse JSON with multiple strategies
            parsed = None
            
            # Strategy 1: Direct parse
            if content and len(content.strip()) > 10:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        logger.info(f"Successfully parsed JSON on attempt {attempt + 1}")
                        return parsed
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Direct parse failed: {e}")
                    pass
            
            # Strategy 2: Fix common issues
            if content and len(content.strip()) > 10:
                fixes = [
                    lambda x: re.sub(r',(\s*[}\]])', r'\1', x),  # Remove trailing commas
                    lambda x: re.sub(r'(["\'])([^"\']*)\1\s*:', r'"\2":', x),  # Fix unquoted keys
                    lambda x: x.replace('\n', ' ').replace('\r', ''),  # Remove newlines
                    lambda x: re.sub(r'\\"', '"', x),  # Fix escaped quotes
                    lambda x: re.sub(r'([^\\])"([^":,}\]]*)"([^:,}\]]*):', r'\1"\2"\3:', x),  # Fix malformed keys
                ]
                
                for fix in fixes:
                    try:
                        fixed = fix(content)
                        if fixed and len(fixed.strip()) > 10:
                            parsed = json.loads(fixed)
                            if isinstance(parsed, dict):
                                logger.info(f"Successfully parsed JSON after fix on attempt {attempt + 1}")
                                return parsed
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
            
            # Strategy 3: Extract fields manually with regex (always succeeds)
            if not parsed:
                logger.warning(f"Attempt {attempt + 1} failed, using regex extraction as fallback")
                
                # Try multiple regex patterns for each field
                question_match = None
                for pattern in [
                    r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    r'question["\']?\s*:\s*["\']([^"\']+)["\']',
                    r'question["\']?\s*:\s*([^,}\]]+)',
                ]:
                    question_match = re.search(pattern, content, re.IGNORECASE)
                    if question_match:
                        break
                
                submit_url_match = None
                for pattern in [
                    r'"submit_url"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    r'submit[_\s]?url["\']?\s*:\s*["\']([^"\']+)["\']',
                    r'https?://[^\s"\'<>]+',
                ]:
                    submit_url_match = re.search(pattern, content, re.IGNORECASE)
                    if submit_url_match:
                        break
                
                task_type_match = re.search(r'"task_type"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
                answer_type_match = re.search(r'"expected_answer_type"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
                
                # Extract data sources
                data_sources = []
                for pattern in [
                    r'"data_sources"\s*:\s*\[(.*?)\]',
                    r'data[_\s]?sources["\']?\s*:\s*\[(.*?)\]',
                ]:
                    sources_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if sources_match:
                        sources_text = sources_match.group(1)
                        url_matches = re.findall(r'https?://[^\s"\'<>]+', sources_text)
                        if url_matches:
                            data_sources = url_matches
                            break
                
                # Build parsed dict with safe defaults
                question_text = ""
                if question_match:
                    question_text = question_match.group(1)
                elif decoded_text:
                    # Extract first sentence from decoded text
                    first_line = decoded_text.split('\n')[0][:200]
                    question_text = first_line
                else:
                    question_text = content[:200] if content else "Unknown question"
                
                submit_url_text = ""
                if submit_url_match:
                    submit_url_text = submit_url_match.group(1) if submit_url_match.groups() else submit_url_match.group(0)
                else:
                    # Look for any URL in content
                    url_match = re.search(r'https?://[^\s"\'<>]+', content)
                    if url_match:
                        submit_url_text = url_match.group(0)
                
                parsed = {
                    "question": question_text.strip(),
                    "submit_url": submit_url_text.strip(),
                    "data_sources": data_sources,
                    "task_type": task_type_match.group(1) if task_type_match else "unknown",
                    "expected_answer_type": answer_type_match.group(1) if answer_type_match else "string",
                    "instructions": (decoded_text[:1000] if decoded_text else content[:1000]).strip()
                }
                
                logger.info(f"Extracted fields using regex on attempt {attempt + 1}")
                return parsed
            
            # If we get here, retry with a simpler prompt
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying with simpler prompt...")
                prompt = f"""Extract from this quiz page:
1. The question/task
2. The submit URL (where to post the answer)

Content: {content_to_parse[:3000]}

Return JSON: {{"question": "...", "submit_url": "...", "data_sources": [], "task_type": "unknown", "expected_answer_type": "string", "instructions": "..."}}"""
                continue
            else:
                # Last attempt failed, return best guess
                logger.error("All parsing attempts failed, using fallback")
                return {
                    "question": decoded_text[:500] if decoded_text else content[:500],
                    "submit_url": "",
                    "data_sources": [],
                    "task_type": "unknown",
                    "expected_answer_type": "string",
                    "instructions": decoded_text[:1000] if decoded_text else content[:1000]
                }
        
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                # Final fallback
                logger.error("All retries exhausted, using emergency fallback")
                return {
                    "question": decoded_text[:500] if decoded_text else html_content[:500],
                    "submit_url": "",
                    "data_sources": [],
                    "task_type": "unknown",
                    "expected_answer_type": "string",
                    "instructions": decoded_text[:1000] if decoded_text else html_content[:1000]
                }
            await asyncio.sleep(1)  # Wait before retry


async def solve_quiz_task(
    question: str,
    data: Optional[Any] = None,
    data_type: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None
) -> Any:
    """
    Use LLM to solve a quiz task given the question and data.
    
    Args:
        question: The quiz question or task
        data: The data to analyze (can be text, DataFrame summary, etc.)
        data_type: Type of data (csv, pdf, text, etc.)
        context: Additional context about the task
    
    Returns:
        The answer (type depends on question)
    """
    if not client:
        raise ValueError("OpenRouter API key not configured")
    
    # Select appropriate model based on task type
    model_id = await select_best_model(task_type or "analysis")
    
    system_prompt = """You are an expert data analyst and problem solver. 
Analyze the given data and question carefully, then provide ONLY the answer value requested.
- For numerical answers, provide ONLY the number (e.g., 12345, not "12345" or {"answer": 12345})
- For boolean answers, provide true or false
- For string answers, provide the exact string value
- For JSON answers, provide ONLY the JSON object value (not wrapped in email/secret/url/submit_url fields)
CRITICAL: Do NOT include email, secret, url, submit_url, or any other metadata fields.
Provide ONLY the answer value itself. If the question asks for JSON, return only the JSON object, not wrapped in any other structure."""

    user_prompt = f"Question: {question}\n\n"
    
    # Handle audio data for transcription
    audio_data = None
    if data is not None and isinstance(data, dict) and data.get("type") == "audio":
        audio_data = data.get("data")  # base64 encoded audio
        task_type = "transcription"
        model_id = await select_best_model("transcription")
    
    if data is not None:
        if isinstance(data, dict) and data.get("type") == "audio":
            # Audio will be sent as base64 in the request
            pass  # Handled separately below
        elif data_type == "csv" or isinstance(data, str):
            # Limit data size to save tokens and costs
            data_str = str(data)
            if len(data_str) > 3000:  # Reduced from 5000 to save costs
                user_prompt += f"Data summary (first 3000 chars): {data_str[:3000]}\n\n"
            else:
                user_prompt += f"Data:\n{data_str}\n\n"
        else:
            # For DataFrames, provide summary
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    user_prompt += f"Data summary:\n{data.head(20).to_string()}\n"
                    user_prompt += f"Shape: {data.shape}\n"
                    user_prompt += f"Columns: {list(data.columns)}\n\n"
                else:
                    user_prompt += f"Data: {str(data)[:3000]}\n\n"
            except:
                user_prompt += f"Data: {str(data)[:3000]}\n\n"
    
    if context:
        user_prompt += f"Context: {context}\n\n"
    
    user_prompt += "Provide the exact answer to the question. Be precise."
    
    # Estimate cost
    estimated_tokens = len(user_prompt.split()) * 1.3
    estimated_cost = estimate_cost(int(estimated_tokens), 1000, model_id)
    
    if not check_budget(estimated_cost):
        raise ValueError(f"Estimated cost ${estimated_cost:.4f} exceeds budget")
    
    try:
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # For audio transcription, include audio data
        # OpenRouter Whisper models accept audio via file upload or base64
        # We'll include it in the prompt for now - OpenRouter will handle the format
        if audio_data:
            # Add audio data reference to the prompt
            user_prompt_with_audio = user_prompt + f"\n\nAudio data (base64): {audio_data[:100]}... (truncated for display)"
            messages[1]["content"] = user_prompt_with_audio
        
        response = await client.post(
            "/chat/completions",
            json={
                "model": model_id,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000  # Reduced to save costs
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Track actual cost
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        actual_cost = estimate_cost(prompt_tokens, completion_tokens, model_id)
        add_cost(actual_cost)
        
        answer = result["choices"][0]["message"]["content"].strip()
        logger.info(f"LLM generated answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error solving quiz task: {e}")
        raise


async def close_client():
    """Close the HTTP client."""
    global client
    if client:
        await client.aclose()
        client = None

