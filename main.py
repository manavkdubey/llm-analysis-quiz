"""FastAPI server for LLM Analysis Quiz endpoint."""
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import uvicorn
import json

from config import EMAIL, SECRET, HOST, PORT
from models import QuizRequest, QuizResponse
from browser import BrowserManager
from quiz_solver import QuizSolver
from llm_client import close_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global browser manager
browser_manager: BrowserManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global browser_manager
    
    # Startup
    logger.info("Starting browser manager...")
    browser_manager = BrowserManager()
    logger.info("Browser manager started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down browser manager...")
    if browser_manager:
        await browser_manager.__aexit__(None, None, None)
    logger.info("Closing LLM client...")
    await close_client()
    logger.info("Shutdown complete")


app = FastAPI(
    title="LLM Analysis Quiz Solver",
    description="API endpoint for solving LLM Analysis Quiz tasks",
    lifespan=lifespan
)


@app.post("/quiz", response_model=dict)
async def solve_quiz(request: QuizRequest):
    """
    Main endpoint to receive quiz tasks and solve them.
    
    Expected payload:
    {
        "email": "your-email@example.com",
        "secret": "your-secret",
        "url": "https://example.com/quiz-123"
    }
    """
    # Verify secret
    if request.secret != SECRET:
        logger.warning(f"Invalid secret provided: {request.secret}")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    # Verify email (optional, but good practice)
    if request.email != EMAIL:
        logger.warning(f"Email mismatch: {request.email} != {EMAIL}")
        # Still allow if secret is correct, but log it
    
    logger.info(f"Received quiz request for URL: {request.url}")
    
    try:
        # Create solver and solve quiz
        # Initialize browser manager if not already done
        if browser_manager is None:
            browser_manager = BrowserManager()
        
        solver = QuizSolver(browser_manager)
        
        # Run solver in background task to avoid blocking
        # But we need to wait for it to complete
        await solver.solve_quiz(request.url)
        
        await solver.close()
        
        return {
            "status": "completed",
            "message": "Quiz solved successfully"
        }
    
    except Exception as e:
        logger.error(f"Error solving quiz: {e}", exc_info=True)
        # Still return 200 to avoid retries, but log the error
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/quiz-sync", response_model=dict)
async def solve_quiz_sync(request: QuizRequest):
    """
    Synchronous version that runs solver in background.
    Returns immediately after starting the solver.
    """
    # Verify secret
    if request.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    logger.info(f"Received quiz request (async) for URL: {request.url}")
    
    # Run in background
    async def solve_in_background():
        try:
            solver = QuizSolver(browser_manager)
            await solver.solve_quiz(request.url)
            await solver.close()
        except Exception as e:
            logger.error(f"Background quiz solving error: {e}", exc_info=True)
    
    asyncio.create_task(solve_in_background())
    
    return {
        "status": "started",
        "message": "Quiz solving started in background"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "email": EMAIL,
        "browser_ready": browser_manager is not None
    }


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors - return 400 for invalid JSON, 422 for missing fields."""
    errors = exc.errors()
    
    # Check if it's a JSON decode error
    for error in errors:
        if error.get("type") == "json_invalid":
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid JSON"}
            )
    
    # For other validation errors, return 422
    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="info"
    )

