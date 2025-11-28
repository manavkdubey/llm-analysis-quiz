# Quick Deployment Checklist

## Before Deploying

1. ✅ All test files removed
2. ✅ Only essential code files remain
3. ✅ Vercel configuration ready
4. ✅ Environment variables documented

## Files That Will Be Visible (Public Repo)

### Core Code (Required)
- main.py
- quiz_solver.py
- browser.py
- llm_client.py
- data_processor.py
- models.py
- config.py

### Config Files
- requirements.txt
- vercel.json
- api/index.py
- .gitignore

### Documentation
- README.md
- DEPLOY.md
- LICENSE

## Steps to Deploy

1. **Push to GitHub** (make it public):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy to Vercel**:
   - Go to https://vercel.com/new
   - Import GitHub repository
   - Add environment variables:
     - `OPENROUTER_API_KEY`
     - `EMAIL=23f2001947@ds.study.iitm.ac.in`
     - `SECRET=manavkumardubey`
   - Build Command: `pip install -r requirements.txt && playwright install chromium`
   - Deploy

3. **Get Your Endpoint**:
   - Vercel will give you: `https://your-project.vercel.app`
   - Your quiz endpoint: `https://your-project.vercel.app/quiz`

4. **Test**:
   ```bash
   curl -X POST https://your-project.vercel.app/quiz \
     -H "Content-Type: application/json" \
     -d '{
       "email": "23f2001947@ds.study.iitm.ac.in",
       "secret": "manavkumardubey",
       "url": "https://tds-llm-analysis.s-anand.net/demo"
     }'
   ```

## Important Notes

⚠️ **Playwright on Vercel**: 
- Browser binaries are large (~300MB)
- May need Vercel Pro plan for better limits
- Alternative: Use Railway or Render for better Playwright support

⚠️ **Environment Variables**:
- Never commit `.env` file
- Set all variables in Vercel dashboard
- Keep API keys secret

## For Google Form Submission

You'll need:
1. **API Endpoint URL**: `https://your-project.vercel.app/quiz`
2. **GitHub Repo URL**: `https://github.com/your-username/your-repo`
3. **System Prompt** (max 100 chars): Create one that resists code word revelation
4. **User Prompt** (max 100 chars): Create one that overrides system prompt
