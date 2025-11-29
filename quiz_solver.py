"""Main quiz solving logic."""
import asyncio
import logging
import re
import json
import httpx
import pandas as pd
import tempfile
import os
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import base64

from browser import BrowserManager
from llm_client import parse_quiz_instructions, solve_quiz_task
from data_processor import (
    download_file, process_pdf, process_csv, process_excel,
    analyze_data, create_visualization
)
from config import EMAIL, SECRET, QUIZ_TIMEOUT_SECONDS
from models import AnswerPayload, QuizResponse

logger = logging.getLogger(__name__)


class QuizSolver:
    """Main class for solving quiz tasks."""
    
    def __init__(self, browser_manager: BrowserManager):
        self.browser = browser_manager
        self.client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        self.start_time: Optional[datetime] = None
    
    def _check_timeout(self):
        """Check if we've exceeded the timeout."""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > QUIZ_TIMEOUT_SECONDS:
                raise TimeoutError(f"Quiz timeout exceeded ({QUIZ_TIMEOUT_SECONDS}s)")
    
    def _has_time_for_retry(self) -> bool:
        """Check if we have enough time for a retry."""
        if not self.start_time:
            return True
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining = QUIZ_TIMEOUT_SECONDS - elapsed
        return remaining > 30
    
    async def extract_quiz_content(self, html_content: str) -> Dict[str, Any]:
        """
        Extract quiz content from HTML, handling base64 encoded content.
        
        Args:
            html_content: HTML content from quiz page
        
        Returns:
            Dictionary with extracted quiz information
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for script tags that might contain base64 encoded content
        scripts = soup.find_all('script')
        decoded_content = ""
        
        for script in scripts:
            script_text = script.string or ""
            # Look for atob() calls with base64 strings
            atob_match = re.search(r'atob\(`([^`]+)`\)', script_text)
            if atob_match:
                try:
                    base64_str = atob_match.group(1)
                    decoded = base64.b64decode(base64_str).decode('utf-8')
                    decoded_content += decoded + "\n"
                except Exception as e:
                    logger.warning(f"Failed to decode base64: {e}")
        
        # Also get text from result divs
        result_divs = soup.find_all(id=re.compile(r'result', re.I))
        for div in result_divs:
            decoded_content += div.get_text() + "\n"
        
        # Combine with full HTML for LLM parsing
        full_content = decoded_content + "\n\n" + html_content
        
        return {
            "decoded_text": decoded_content,
            "full_html": full_content,
            "soup": soup
        }
    
    async def fetch_data_source(self, url: str, headers: Optional[Dict] = None) -> Any:
        """
        Fetch data from a source (file, API, etc.).
        
        Args:
            url: URL to fetch from
            headers: Optional HTTP headers
        
        Returns:
            Processed data (DataFrame, text, etc.)
        """
        self._check_timeout()
        
        logger.info(f"Fetching data from {url}")
        
        # Determine file type from URL or content
        if url.endswith('.pdf'):
            content = download_file(url, headers)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                f.write(content)
                temp_path = f.name
            try:
                return process_pdf(temp_path)
            finally:
                os.unlink(temp_path)
        
        elif url.endswith('.csv'):
            content = download_file(url, headers)
            return process_csv(content)
        
        elif url.endswith(('.xlsx', '.xls')):
            content = download_file(url, headers)
            return process_excel(content)
        
        elif url.endswith(('.opus', '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac')):
            # Audio files - return as binary for transcription
            content = download_file(url, headers)
            return {"type": "audio", "data": base64.b64encode(content).decode('utf-8'), "url": url}
        
        else:
            # Try as API or text
            response = await self.client.get(url, headers=headers or {})
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            
            if 'json' in content_type:
                return response.json()
            elif 'csv' in content_type:
                return process_csv(response.text)
            elif 'audio' in content_type:
                # Audio content - return as binary for transcription
                audio_data = response.content
                return {"type": "audio", "data": base64.b64encode(audio_data).decode('utf-8'), "url": url}
            else:
                return response.text
    
    async def solve_quiz(self, quiz_url: str, max_retries: int = 2) -> None:
        """
        Main method to solve a quiz recursively.
        
        Args:
            quiz_url: URL of the quiz to solve
            max_retries: Maximum number of attempts to solve this quiz
        """
        if self.start_time is None:
            self.start_time = datetime.now()
        
        self._check_timeout()
        
        logger.info(f"Solving quiz at {quiz_url}")
        
        try:
            html_content = await self.browser.get_page_content(quiz_url)
            extracted = await self.extract_quiz_content(html_content)
            
            from urllib.parse import urlparse, parse_qs, urlencode
            if "personalized" in html_content.lower() and "email" not in urlparse(quiz_url).query:
                parsed = urlparse(quiz_url)
                params = parse_qs(parsed.query)
                params['email'] = [EMAIL]
                new_query = urlencode(params, doseq=True)
                quiz_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"
                html_content = await self.browser.get_page_content(quiz_url)
                extracted = await self.extract_quiz_content(html_content)
            
            # Step 2: Parse instructions using LLM (with retries)
            try:
                instructions = await parse_quiz_instructions(extracted["full_html"], max_retries=3)
            except Exception as e:
                logger.error(f"Failed to parse instructions after retries: {e}")
                instructions = {
                    "question": extracted["decoded_text"][:500] if extracted["decoded_text"] else "Unknown question",
                    "submit_url": "",
                    "data_sources": [],
                    "task_type": "unknown",
                    "expected_answer_type": "string",
                    "instructions": extracted["decoded_text"][:1000] if extracted["decoded_text"] else ""
                }
            question = instructions.get("question", "")
            submit_url = instructions.get("submit_url", "")
            data_sources = instructions.get("data_sources", [])
            task_type = instructions.get("task_type", "unknown")
            expected_type = instructions.get("expected_answer_type", "string")
            
            if not submit_url:
                from urllib.parse import urlparse
                parsed = urlparse(quiz_url)
                submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
            
            if submit_url and not submit_url.startswith('http'):
                from urllib.parse import urljoin
                submit_url = urljoin(quiz_url, submit_url)
            
            logger.info(f"Question: {question}")
            logger.info(f"Submit URL: {submit_url}")
            logger.info(f"Data sources: {data_sources}")
            
            if not question or "how to play" in question.lower() or "start by posting" in question.lower():
                logger.info("Detected instructions page, submitting initial answer to start quiz chain")
                if "project2" in quiz_url:
                    from urllib.parse import urlparse
                    parsed = urlparse(quiz_url)
                    initial_answer = f"{parsed.path}"
                else:
                    initial_answer = ""
                response = await self._submit_answer(submit_url, quiz_url, initial_answer)
                if response.url:
                    from urllib.parse import urlparse, urljoin, parse_qs, urlencode
                    next_url = response.url
                    if not next_url.startswith('http'):
                        parsed = urlparse(quiz_url)
                        next_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", next_url)
                    parsed_next = urlparse(next_url)
                    if "personalized" in next_url.lower() or "email" in str(response).lower():
                        params = parse_qs(parsed_next.query)
                        if 'email' not in params:
                            params['email'] = [EMAIL]
                            new_query = urlencode(params, doseq=True)
                            next_url = f"{parsed_next.scheme}://{parsed_next.netloc}{parsed_next.path}?{new_query}"
                    await self.solve_quiz(next_url)
                return
            
            # Step 3: Fetch and process data if needed
            data = None
            if data_sources:
                source_url = data_sources[0]
                if not source_url.startswith('http'):
                    from urllib.parse import urljoin
                    source_url = urljoin(quiz_url, source_url)
                data = await self.fetch_data_source(source_url)
            
            # Step 4-5: Try to solve and submit (with retries)
            attempt = 0
            while attempt < max_retries:
                attempt += 1
                self._check_timeout()
                
                # Step 4: Solve the task
                answer = await self._solve_task(
                    question=question,
                    data=data,
                    task_type=task_type,
                    expected_type=expected_type,
                    instructions=instructions
                )
                
                # Extract just the answer value if LLM returned a full payload structure
                if isinstance(answer, str):
                    # Try to parse as JSON first
                    try:
                        parsed = json.loads(answer)
                        if isinstance(parsed, dict):
                            # Remove metadata keys, keep only answer
                            metadata_keys = ['email', 'secret', 'url', 'submit_url']
                            if 'answer' in parsed:
                                answer = parsed['answer']
                            else:
                                # If no 'answer' key, use the first non-metadata value
                                non_metadata = {k: v for k, v in parsed.items() if k not in metadata_keys}
                                if non_metadata:
                                    answer = next(iter(non_metadata.values()))
                                else:
                                    # Empty or only metadata - this is invalid, retry
                                    logger.warning(f"LLM returned invalid response: {answer}, retrying...")
                                    if attempt < max_retries:
                                        continue
                                    answer = ""  # Fallback
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, check if it's a quoted string that needs unquoting
                        answer = answer.strip().strip('"').strip("'")
                
                if isinstance(answer, dict):
                    # Remove metadata keys
                    metadata_keys = ['email', 'secret', 'url', 'submit_url']
                    if 'answer' in answer:
                        answer = answer['answer']
                    else:
                        # Get first non-metadata value
                        non_metadata = {k: v for k, v in answer.items() if k not in metadata_keys}
                        if non_metadata:
                            answer = next(iter(non_metadata.values()))
                        else:
                            # Empty dict or only metadata - invalid, retry
                            logger.warning(f"LLM returned empty/invalid dict: {answer}, retrying...")
                            if attempt < max_retries:
                                continue
                            answer = ""  # Fallback
                
                # If answer is still empty or invalid, retry
                if answer == "" or answer == {} or (isinstance(answer, dict) and len(answer) == 0):
                    logger.warning(f"Empty answer detected (attempt {attempt}/{max_retries})")
                    if attempt < max_retries:
                        continue
                
                # Convert answer to proper type
                answer = self._convert_answer_type(answer, expected_type)
                
                logger.info(f"Final answer (attempt {attempt}, type: {type(answer).__name__}): {answer}")
                
                # Step 5: Submit answer
                response = await self._submit_answer(submit_url, quiz_url, answer)
                
                # Step 6: Handle response
                if response.correct:
                    logger.info("Answer is correct!")
                    if response.url:
                        from urllib.parse import urlparse, urljoin, parse_qs, urlencode
                        next_url = response.url
                        if not next_url.startswith('http'):
                            parsed = urlparse(quiz_url)
                            next_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", next_url)
                        parsed_next = urlparse(next_url)
                        if "personalized" in next_url.lower() or "email" in str(response).lower():
                            params = parse_qs(parsed_next.query)
                            if 'email' not in params:
                                params['email'] = [EMAIL]
                                new_query = urlencode(params, doseq=True)
                                next_url = f"{parsed_next.scheme}://{parsed_next.netloc}{parsed_next.path}?{new_query}"
                        await self.solve_quiz(next_url)
                    else:
                        logger.info("Quiz completed!")
                    return
                else:
                    logger.warning(f"Answer incorrect (attempt {attempt}): {response.reason}")
                    
                    if attempt < max_retries and self._has_time_for_retry():
                        logger.info(f"Retrying with different approach (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(1)
                        continue
                    elif response.url:
                        logger.info("Moving to next quiz URL")
                        from urllib.parse import urlparse, urljoin, parse_qs, urlencode
                        next_url = response.url
                        if not next_url.startswith('http'):
                            parsed = urlparse(quiz_url)
                            next_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", next_url)
                        parsed_next = urlparse(next_url)
                        if "personalized" in next_url.lower() or "email" in str(response).lower():
                            params = parse_qs(parsed_next.query)
                            if 'email' not in params:
                                params['email'] = [EMAIL]
                                new_query = urlencode(params, doseq=True)
                                next_url = f"{parsed_next.scheme}://{parsed_next.netloc}{parsed_next.path}?{new_query}"
                        await self.solve_quiz(next_url)
                        return
                    elif response.url:
                        logger.info("Moving to next quiz URL")
                        await self.solve_quiz(response.url)
                        return
                    else:
                        logger.error("No more retries and no next URL provided")
                        return
        
        except TimeoutError:
            logger.error("Quiz timeout exceeded")
            raise
        except Exception as e:
            logger.error(f"Error solving quiz: {e}", exc_info=True)
            raise
    
    async def _solve_task(
        self,
        question: str,
        data: Any,
        task_type: str,
        expected_type: str,
        instructions: Dict[str, Any]
    ) -> Union[bool, int, float, str, Dict[str, Any]]:
        """
        Solve a specific task based on type.
        
        Args:
            question: The question to answer
            data: The data to work with
            task_type: Type of task
            expected_type: Expected answer type
        
        Returns:
            The answer in the appropriate format
        """
        self._check_timeout()
        
        if task_type == "data_analysis" and data is not None:
            # Try to extract analysis operation from question
            if isinstance(data, pd.DataFrame):
                # Look for specific operations in question
                if "sum" in question.lower():
                    # Extract column name if mentioned
                    column_match = re.search(r'["\'](\w+)["\']', question)
                    column = column_match.group(1) if column_match else None
                    result = analyze_data(data, "sum", column=column)
                    return self._convert_answer_type(result, expected_type)
                
                elif "mean" in question.lower() or "average" in question.lower():
                    column_match = re.search(r'["\'](\w+)["\']', question)
                    column = column_match.group(1) if column_match else None
                    result = analyze_data(data, "mean", column=column)
                    return self._convert_answer_type(result, expected_type)
                
                elif "count" in question.lower():
                    result = analyze_data(data, "count")
                    return self._convert_answer_type(result, expected_type)
        
        # Use LLM for complex tasks
        answer = await solve_quiz_task(
            question, 
            data, 
            type(data).__name__ if data is not None else None, 
            instructions,
            task_type=task_type
        )
        
        return answer
    
    def _convert_answer_type(self, answer: Any, expected_type: str) -> Union[bool, int, float, str, Dict[str, Any]]:
        """Convert answer to expected type."""
        if answer is None:
            return "" if expected_type != "json" else {}
        
        if expected_type == "boolean":
            if isinstance(answer, bool):
                return answer
            answer_str = str(answer).lower().strip()
            return answer_str in ("true", "1", "yes", "correct")
        
        elif expected_type == "number":
            if isinstance(answer, (int, float)):
                return answer
            if isinstance(answer, str):
                numbers = re.findall(r'-?\d+\.?\d*', answer)
                if numbers:
                    try:
                        return float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                    except:
                        pass
            return 0
        
        elif expected_type == "json":
            if isinstance(answer, dict):
                # If it's already a dict, return as is
                # But if it has nested structure that might cause issues, flatten it
                if len(answer) == 1 and 'answer' in answer:
                    return answer['answer']
                return answer
            if isinstance(answer, str):
                answer = answer.strip()
                if answer.startswith('{') or answer.startswith('['):
                    try:
                        parsed = json.loads(answer)
                        if isinstance(parsed, dict):
                            # If parsed dict has 'answer' key, extract it
                            if len(parsed) == 1 and 'answer' in parsed:
                                return parsed['answer']
                            return parsed
                    except:
                        pass
                return answer
            return str(answer)
        
        else:  # string or base64
            return str(answer)
    
    async def _submit_answer(self, submit_url: str, quiz_url: str, answer: Any) -> QuizResponse:
        """
        Submit answer to the quiz endpoint.
        
        Args:
            submit_url: URL to submit to
            quiz_url: Original quiz URL
            answer: The answer to submit
        
        Returns:
            QuizResponse with result
        """
        self._check_timeout()
        
        # Ensure submit_url is absolute
        if not submit_url.startswith('http'):
            from urllib.parse import urljoin
            submit_url = urljoin(quiz_url, submit_url)
        
        payload = AnswerPayload(
            email=EMAIL,
            secret=SECRET,
            url=quiz_url,
            answer=answer
        )
        
        payload_dict = payload.dict()
        logger.info(f"Submitting answer to {submit_url}")
        
        try:
            response = await self.client.post(
                submit_url,
                json=payload_dict,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            
            if response.status_code != 200:
                error_text = response.text[:500]
                logger.error(f"Submit failed with status {response.status_code}: {error_text}")
                response.raise_for_status()
            
            result = response.json()
            logger.info(f"Submit response: {result}")
            return QuizResponse(**result)
        
        except httpx.HTTPStatusError as e:
            error_text = ""
            if hasattr(e, 'response') and e.response:
                error_text = e.response.text[:500]
            logger.error(f"HTTP error submitting answer: {e.response.status_code if hasattr(e, 'response') else 'unknown'} - {error_text}")
            raise
        except Exception as e:
            logger.error(f"Error submitting answer: {e}")
            raise
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()

