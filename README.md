# Today in History

Production backend that fetches historical events from Wikipedia, generates AI summaries via Gemini, stores them in Supabase, and serves them through REST APIs.

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **Database:** Supabase (PostgreSQL)
- **AI:** Google Gemini API
- **Scheduler:** APScheduler
- **HTTP Client:** httpx

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/events/today` | Today's historical events |
| `GET` | `/api/events/{month}/{day}` | Events by date |
| `POST` | `/api/events/trigger/{month}/{day}` | Manually trigger pipeline |
| `GET` | `/docs` | Swagger UI |

**Query params:** `?skip=0&limit=20&year=1969`

## Local Setup

```bash
# Clone
git clone https://github.com/anuragsingh6436/today-in-history.git
cd today-in-history

# Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Config
cp .env.example .env
# Edit .env with your keys

# Database
# Run schema.sql in Supabase SQL Editor

# Run
uvicorn app.main:app --reload

# Test
pytest tests/ -v
```

## Deploy to Railway

### Step 1: Supabase Setup
1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run `schema.sql`
3. Copy your **Project URL** and **service_role key** from Settings > API

### Step 2: Gemini API Key
1. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Step 3: Railway Deploy
1. Go to [railway.app](https://railway.app) and create a new project
2. Select **Deploy from GitHub repo** → pick `today-in-history`
3. Go to **Variables** tab and add:

| Variable | Value |
|----------|-------|
| `SUPABASE_URL` | `https://your-project.supabase.co` |
| `SUPABASE_KEY` | Your service_role key |
| `GEMINI_API_KEY` | Your Gemini API key |
| `SCHEDULER_ENABLED` | `true` |
| `SCHEDULER_HOUR_UTC` | `0` (midnight UTC) |
| `CORS_ORIGINS` | Your frontend URL (or `*`) |

4. Railway auto-detects `Procfile` and deploys
5. Health check at `https://your-app.railway.app/health`

### Step 4: Verify
```bash
# Check health
curl https://your-app.railway.app/health

# Trigger pipeline manually
curl -X POST https://your-app.railway.app/api/events/trigger/4/11

# Fetch events
curl https://your-app.railway.app/api/events/today
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SUPABASE_URL` | Yes | — | Supabase project URL |
| `SUPABASE_KEY` | Yes | — | Supabase service role key |
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model ID |
| `GEMINI_MAX_RETRIES` | No | `3` | Max retry attempts |
| `GEMINI_BASE_DELAY` | No | `1.0` | Backoff base delay (seconds) |
| `DEBUG` | No | `false` | Enable debug logging |
| `CORS_ORIGINS` | No | `*` | Comma-separated origins |
| `SCHEDULER_ENABLED` | No | `true` | Enable daily scheduler |
| `SCHEDULER_HOUR_UTC` | No | `0` | Daily job hour (UTC) |
| `SCHEDULER_MINUTE_UTC` | No | `0` | Daily job minute (UTC) |

## Project Structure

```
app/
├── main.py              # App factory, lifespan, middleware
├── config.py            # Settings via pydantic-settings
├── db/supabase.py       # Database abstraction layer
├── models/
│   ├── event.py         # Domain models
│   ├── pipeline.py      # Pipeline result model
│   └── response.py      # API response envelopes
├── routes/events.py     # REST endpoints
├── services/
│   ├── wikipedia.py     # Wikipedia API fetcher
│   ├── gemini.py        # AI summarization + retry
│   ├── prompts.py       # Configurable prompt templates
│   └── pipeline.py      # Fetch → AI → Store orchestrator
└── scheduler/jobs.py    # APScheduler daily job
```
