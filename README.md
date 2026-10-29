# agent-code-manage

Give it a GitHub repository and it writes a review of the whole codebase: a general overview, or a focused report on security, architecture, code quality or performance.

It is a small FastAPI service with no database. One request in, one Markdown report out. The [code-bot](https://github.com/Rakshit-gen/code-bot) UI uses it for its "analyze a repository" page; [agent-prm](https://github.com/Rakshit-gen/agent-prm) is the sibling service that reviews single pull requests.

## How it works

For each `POST /analyze`:

1. **Read the repo through the GitHub API.**
   - Repo metadata: description, stars, main language, default branch.
   - A file tree three levels deep (the first 100 entries go into the prompt).
   - Up to 50 source files under 100 KB, found breadth-first. Sixteen languages are recognised by extension (`.py`, `.ts`, `.go`, `.rs`, `.java` and so on).
   - Those files are sorted so entry points come first (`main.`, `app.`, `index.`, `server.`), then routes, controllers, models, services, components. The top 15 are read, 2,000 characters each.
   - The README, first 2,000 characters.
2. **Build one prompt** from all of that plus a template for the chosen report type. Each template asks for fixed sections, for example the security one asks for critical vulnerabilities, auth, data protection, API security, dependencies and recommendations.
3. **Call Groq.** It tries `llama-3.3-70b-versatile` first. If Groq says rate limited, it moves down a list of smaller models. Any other error stops right away.
4. **Return** the report text with the repo name, report type, number of files read, stars and language.

## Quickstart

You need Python 3.11+, a [Groq API key](https://console.groq.com/keys) and a [GitHub token](https://github.com/settings/tokens). Both keys are required; the service will not start without them.

```bash
git clone https://github.com/Rakshit-gen/agent-code-manage.git
cd agent-code-manage
pip install -r requirements.txt
cat > .env <<'EOF'
GROQ_API_KEY=gsk_...
GITHUB_TOKEN=ghp_...
EOF
uvicorn main:app --reload
```

Then:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"repository": "facebook/react", "type": "architecture"}'
```

`repository` takes `owner/repo` or a full GitHub URL. `type` is one of `general` (default), `security`, `architecture`, `quality`, `performance`.

Interactive docs are at `/docs`.

## Configuration

| Variable | Required | What it does |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq key |
| `GITHUB_TOKEN` | yes | Reads repo contents. Give it `repo` scope if you want private repos |
| `PORT` | no | Only used by `python main.py`, default 8000 |

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/analyze` | Runs a review and returns it. Takes 10 to 60 seconds depending on repo size and Groq load |
| `POST` | `/webhook` | GitHub push webhook. A push to `main` or `master` starts a general review in the background |
| `GET` | `/health` | Says whether both keys are set |
| `GET` | `/` | Service info and supported report types |

Response:

```json
{
  "success": true,
  "repository": "facebook/react",
  "analysis_type": "architecture",
  "analysis": "## System Design\n...",
  "files_analyzed": 15,
  "stars": 230000,
  "language": "JavaScript",
  "metadata": {"default_branch": "main"}
}
```

## Deploy on Render

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Set `GROQ_API_KEY` and `GITHUB_TOKEN` in the dashboard.

## Limits

- It sees at most 15 files and 2,000 characters of each. On a big repo the report describes a sample, not the whole thing.
- One request makes one LLM call per report, and many GitHub API calls to walk the tree. Large repos can hit GitHub's rate limit.
- The webhook does not check GitHub's signature, and the report it produces is only logged, not stored or posted anywhere.
- There is no auth or rate limiting on `/analyze`.
- `docker-compose.yml` and the files under `tests/` came over from agent-prm and still reference Redis and a rate limiter that this service does not have. Use `uvicorn` directly for now.
- Some fallback model names in the list have since been retired by Groq. When they fail with something other than a rate limit, the loop stops.
