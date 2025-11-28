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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

browser_manager = BrowserManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser_manager
    logger.info("Browser manager ready")
    yield
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
    if request.secret != SECRET:
        logger.warning(f"Invalid secret provided: {request.secret}")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    if request.email != EMAIL:
        logger.warning(f"Email mismatch: {request.email} != {EMAIL}")
    
    logger.info(f"Received quiz request for URL: {request.url}")
    
    try:
        solver = QuizSolver(browser_manager)
        await solver.solve_quiz(request.url)
        await solver.close()
        
        return {
            "status": "completed",
            "message": "Quiz solved successfully"
        }
    
    except Exception as e:
        logger.error(f"Error solving quiz: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/quiz-sync", response_model=dict)
async def solve_quiz_sync(request: QuizRequest):
    if request.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    logger.info(f"Received quiz request (async) for URL: {request.url}")
    
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
    return {
        "status": "healthy",
        "email": EMAIL,
        "browser_ready": browser_manager is not None
    }


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    
    for error in errors:
        if error.get("type") == "json_invalid":
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid JSON"}
            )
    
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

