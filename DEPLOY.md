# Deployment Guide for Vercel

## Prerequisites
1. Vercel account (sign up at https://vercel.com)
2. GitHub repository (make it public for evaluation)
3. OpenRouter API key

## Important Notes

⚠️ **Playwright on Vercel**: Playwright requires browser binaries which are large. Vercel has size limits. Consider:
- Using a lighter headless browser solution
- Or deploying to a platform that supports Playwright better (Railway, Render, etc.)

## Deployment Steps

### 1. Prepare Repository
- Ensure all code is committed
- Make repository public (required for evaluation)
- Add MIT LICENSE file

### 2. Set Environment Variables on Vercel
Go to Vercel Dashboard → Your Project → Settings → Environment Variables:

```
OPENROUTER_API_KEY=your_openrouter_api_key
EMAIL=23f2001947@ds.study.iitm.ac.in
SECRET=manavkumardubey
```

### 3. Deploy to Vercel

**Option A: Via Vercel Dashboard**
1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Framework Preset: Other
4. Build Command: (leave empty)
5. Output Directory: (leave empty)
6. Install Command: `pip install -r requirements.txt && playwright install chromium`
7. Click Deploy

**Option B: Via Vercel CLI**
```bash
npm i -g vercel
vercel login
vercel
```

### 4. Configure Build Settings
- Build Command: `pip install -r requirements.txt && playwright install chromium`
- Install Command: (leave empty)
- Output Directory: (leave empty)

### 5. Get Your Endpoint URL
After deployment, Vercel will provide a URL like:
`https://your-project.vercel.app`

Your quiz endpoint will be:
`https://your-project.vercel.app/quiz`

### 6. Test Your Endpoint
```bash
curl -X POST https://your-project.vercel.app/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "23f2001947@ds.study.iitm.ac.in",
    "secret": "manavkumardubey",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

## Alternative: Railway or Render

If Vercel has issues with Playwright, consider:

### Railway
1. Sign up at https://railway.app
2. New Project → Deploy from GitHub
3. Add environment variables
4. Deploy

### Render
1. Sign up at https://render.com
2. New Web Service → Connect GitHub
3. Build Command: `pip install -r requirements.txt && playwright install chromium`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables
6. Deploy

## Troubleshooting

- **Playwright browser not found**: Run `playwright install chromium` in build step
- **Timeout errors**: Increase function timeout in vercel.json (max 300s)
- **Memory issues**: Vercel has memory limits, consider upgrading plan

