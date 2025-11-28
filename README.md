# LLM Analysis Quiz Solver

An automated system for solving data analysis quizzes using LLMs, headless browsers, and data processing capabilities.

## Features

- **Lightweight Browser**: Uses httpx with base64 extraction for JavaScript-rendered quiz pages (Vercel-compatible)
- **LLM-Powered Parsing**: Uses OpenRouter API with intelligent model selection for cost-effective parsing and problem solving
- **Smart Model Selection**: Automatically selects the best cost-effective model based on available options and budget constraints
- **Cost Tracking**: Monitors API usage to stay within budget limits
- **Data Processing**: Supports PDF, CSV, Excel, and various data formats
- **Data Analysis**: Performs statistical operations, filtering, aggregation
- **Visualization**: Generates charts and visualizations as base64 images
- **Recursive Quiz Solving**: Automatically follows quiz chains until completion
- **Timeout Management**: Ensures completion within 3-minute time limit

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For server deployment with Playwright, use `requirements_server.txt` instead.

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:
```
EMAIL=your-email@example.com
SECRET=your-secret-string
OPENROUTER_API_KEY=your-openrouter-api-key
# Optional: Leave LLM_MODEL empty for auto-selection, or specify like "openai/gpt-3.5-turbo"
LLM_MODEL=
HOST=0.0.0.0
PORT=8000
```

**Note**: The system will automatically select cost-effective models from OpenRouter. You can get an API key from [OpenRouter.ai](https://openrouter.ai/). The system is configured with a $5 budget limit and will choose models accordingly.

### 3. Run the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST `/quiz`

Main endpoint for receiving quiz tasks.

**Request:**
```json
{
  "email": "your-email@example.com",
  "secret": "your-secret",
  "url": "https://example.com/quiz-123"
}
```

**Response:**
```json
{
  "status": "completed",
  "message": "Quiz solved successfully"
}
```

### POST `/quiz-sync`

Asynchronous version that returns immediately and solves in background.

### GET `/health`

Health check endpoint.

## Testing

Test with the demo endpoint:

```bash
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "secret": "your-secret",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

## Architecture

- **main.py**: FastAPI server with endpoints
- **quiz_solver.py**: Core quiz solving logic
- **browser.py**: Lightweight browser manager (httpx-based, Vercel-compatible)
- **llm_client.py**: OpenRouter API integration with smart model selection
- **data_processor.py**: Data processing utilities (PDF, CSV, Excel, visualization)
- **models.py**: Pydantic models for validation
- **config.py**: Configuration management

## Deployment

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions.

### Quick Deploy to Vercel

1. Push code to GitHub (make repository public)
2. Go to [Vercel](https://vercel.com) and import your repository
3. Set environment variables:
   - `OPENROUTER_API_KEY`
   - `EMAIL`
   - `SECRET`
4. Deploy

**Note**: The codebase uses a lightweight httpx-based browser for Vercel compatibility. For server deployments, Playwright can be used (see `requirements_server.txt`).

## License

MIT License - See LICENSE file for details.

