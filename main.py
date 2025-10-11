# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests
from groq import Groq
import logging
import json
import redis
import tempfile
import shutil
import subprocess
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codebase-review")

app = FastAPI(title="AI Codebase Review Agent (Repo)")

# CORS: allow your frontend origins here
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:3000",
    "https://code-bot-rho.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Redis connection (optional)
# -------------------------
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None
else:
    logger.info("No REDIS_URL set; using in-memory storage")


# Use Redis for tasks if available, otherwise in-memory dict
tasks_store: Dict[str, Dict[str, Any]] = {}

def save_task(task_id: str, data: Dict[str, Any]):
    if redis_client:
        try:
            redis_client.set(f"task:{task_id}", json.dumps(data), ex=7 * 24 * 3600)  # 7 days
        except Exception as e:
            logger.error(f"Redis save error: {e}")
            tasks_store[task_id] = data
    else:
        tasks_store[task_id] = data

def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    if redis_client:
        try:
            raw = redis_client.get(f"task:{task_id}")
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return tasks_store.get(task_id)
    else:
        return tasks_store.get(task_id)

# -------------------------
# Request model
# -------------------------
class RepoRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    github_token: Optional[str] = None

# -------------------------
# Codebase Review Agent class
# -------------------------
class CodebaseReviewAgent:
    def __init__(self, github_token: Optional[str] = None):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self.groq_client = Groq(api_key=groq_api_key)
        self.github_token = github_token
        self.headers = {}
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

    def clone_repo(self, repo_url: str, branch: str) -> str:
        # shallow clone of branch into temporary dir
        tmp_dir = tempfile.mkdtemp()
        cmd = ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmp_dir]
        logger.info("Cloning %s @ %s", repo_url, branch)
        subprocess.run(cmd, check=True)
        return tmp_dir

    def get_code_files(self, root_path: str) -> List[str]:
        code_files = []
        for root, _, files in os.walk(root_path):
            for f in files:
                if f.endswith((".py", ".js", ".ts", ".jsx", ".tsx")):
                    code_files.append(os.path.join(root, f))
        return code_files

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except Exception as e:
            logger.error("Failed to read file %s: %s", filepath, e)
            return {"file": filepath, "issues": [], "error": f"read_failed: {str(e)}"}

        if not content.strip():
            return {"file": filepath, "issues": []}

        # Compose a concise prompt instructing the model to include line numbers
        prompt = f"""
You are a precise senior code reviewer. Analyze the following file and report ALL issues.
Return ONLY a JSON array (no markdown, no explanations). Each item must contain:
- "file": the filepath (string)
- "line": line number (integer) where the issue occurs, or null if unknown
- "type": one of "bug", "style", "performance", "security", "best_practice"
- "description": short description
- "suggestion": how to fix it

If there are no issues, return [].

File path: {filepath}

Code:
{content}
"""
        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a strict senior code reviewer. Output only a JSON array."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.15,
                max_tokens=1500,
            )
            text = resp.choices[0].message.content.strip()
            # strip possible code fences
            if text.startswith("```json"):
                text = text[len("```json"):].strip()
            if text.startswith("```"):
                # remove leading fence
                text = text[3:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()

            issues = json.loads(text)
            # Normalize each issue to include file and expected keys
            normalized = []
            for it in issues if isinstance(issues, list) else []:
                normalized.append({
                    "file": it.get("file", filepath) if isinstance(it, dict) else filepath,
                    "line": it.get("line") if isinstance(it, dict) else None,
                    "type": it.get("type") if isinstance(it, dict) else None,
                    "description": it.get("description") if isinstance(it, dict) else str(it),
                    "suggestion": it.get("suggestion") if isinstance(it, dict) else None
                })
            return {"file": filepath, "issues": normalized}
        except Exception as e:
            logger.error("Model error for %s: %s", filepath, e)
            # safe fallback: return empty issues + error message
            return {"file": filepath, "issues": [], "error": str(e)}

    def analyze_repo(self, repo_url: str, branch: str) -> Dict[str, Any]:
        repo_path = None
        try:
            repo_path = self.clone_repo(repo_url, branch)
            files = self.get_code_files(repo_path)
            logger.info("Found %d code files", len(files))
            analyzed = []
            total_issues = 0

            for fp in files:
                res = self.analyze_file(fp)
                analyzed.append(res)
                total_issues += len(res.get("issues", []))

            return {
                "repo_url": repo_url,
                "branch": branch,
                "analyzed_at": datetime.utcnow().isoformat(),
                "total_files": len(analyzed),
                "total_issues": total_issues,
                "files": analyzed
            }
        finally:
            if repo_path and os.path.exists(repo_path):
                try:
                    shutil.rmtree(repo_path)
                except Exception:
                    pass

# -------------------------
# Background worker wrapper (keeps same semantics)
# -------------------------
def process_repo_analysis(task_id: str, repo_url: str, branch: str, github_token: Optional[str] = None):
    try:
        task = get_task(task_id)
        if task:
            task["status"] = "processing"
            save_task(task_id, task)

        logger.info("Starting repo analysis task %s", task_id)
        agent = CodebaseReviewAgent(github_token)
        results = agent.analyze_repo(repo_url, branch)

        task = get_task(task_id)
        if task:
            task["status"] = "completed"
            task["results"] = results
            save_task(task_id, task)

        logger.info("Completed repo analysis task %s", task_id)
    except Exception as e:
        logger.exception("Repo analysis failed for %s: %s", task_id, e)
        task = get_task(task_id)
        if task:
            task["status"] = "failed"
            task["error"] = str(e)
            save_task(task_id, task)

# -------------------------
# Routes
# -------------------------
@app.post("/analyze-repo")
async def analyze_repo(request: Request, repo_request: RepoRequest, background_tasks: BackgroundTasks):
    # create task and schedule background job
    task_id = str(uuid.uuid4())
    task_data = {
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "repo_url": repo_request.repo_url,
        "branch": repo_request.branch
    }
    save_task(task_id, task_data)
    # schedule background
    background_tasks.add_task(process_repo_analysis, task_id, repo_request.repo_url, repo_request.branch, repo_request.github_token)
    logger.info("Created repo analysis task %s", task_id)
    return {"task_id": task_id, "status": "pending"}

@app.get("/status/{task_id}")
async def status(task_id: str):
    t = get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "status": t.get("status", "unknown"),
        "created_at": t.get("created_at")
    }

@app.get("/results/{task_id}")
async def results(task_id: str):
    t = get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    status_val = t.get("status")
    if status_val in ("pending", "processing"):
        raise HTTPException(status_code=202, detail=f"Task {status_val}")
    if status_val == "failed":
        raise HTTPException(status_code=500, detail=f"Failed: {t.get('error')}")
    return {"task_id": task_id, "results": t.get("results", {})}

@app.get("/")
async def root():
    return {
        "message": "AI Codebase Review Agent",
        "version": "1.0.0",
        "storage": "Redis" if redis_client else "In-Memory",
        "endpoints": {
            "POST /analyze-repo": "Start full repo review",
            "GET /status/{task_id}": "Get task status",
            "GET /results/{task_id}": "Get results"
        }
    }

# -------------------------
# If run as script, start uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
