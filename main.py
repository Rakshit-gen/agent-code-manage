from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import uuid
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
from groq import Groq
import logging
import json
import redis
import tempfile
import shutil
import subprocess
import time

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codebase-review")

# -------------------------
# FASTAPI INIT
# -------------------------
app = FastAPI(title="AI Codebase Review Agent (Repo)")

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
# REDIS INIT
# -------------------------
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None
else:
    logger.info("No REDIS_URL provided, using in-memory storage")

tasks_store: Dict[str, Dict[str, Any]] = {}


def save_task(task_id: str, data: Dict[str, Any]):
    if redis_client:
        try:
            redis_client.set(f"task:{task_id}", json.dumps(data), ex=7 * 24 * 3600)
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
# REQUEST MODEL
# -------------------------
class RepoRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    github_token: Optional[str] = None


# -------------------------
# MAIN REVIEW AGENT
# -------------------------
class CodebaseReviewAgent:
    def __init__(self, github_token: Optional[str] = None):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.groq_client = Groq(api_key=groq_api_key)
        self.github_token = github_token
        self.headers = {}
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

    def clone_repo(self, repo_url: str, branch: str) -> str:
        tmp_dir = tempfile.mkdtemp()
        cmd = ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmp_dir]
        logger.info(f"Cloning {repo_url}@{branch}")
        subprocess.run(cmd, check=True)
        return tmp_dir

    def get_code_files(self, root_path: str) -> List[str]:
        code_files = []
        for root, _, files in os.walk(root_path):
            for f in files:
                if f.endswith((".py", ".js", ".ts", ".jsx", ".tsx")):
                    if (
                        "__pycache__" not in root
                        and "node_modules" not in root
                        and "test" not in root.lower()
                    ):
                        code_files.append(os.path.join(root, f))
        return code_files

    def analyze_repo(self, repo_url: str, branch: str) -> Dict[str, Any]:
        repo_path = None
        try:
            repo_path = self.clone_repo(repo_url, branch)
            files = self.get_code_files(repo_path)

            # Limit based on repo size
            if len(files) > 50:
                files = files[:25]
            elif len(files) > 25:
                files = files[:20]

            logger.info(f"Analyzing {len(files)} files from repo")

            analyzed = []
            total_issues = 0
            batch_size = 5

            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                combined_code = ""

                for fp in batch:
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        combined_code += f"\n\n# FILE: {fp}\n{content}\n"
                    except Exception as e:
                        logger.warning(f"Skipping {fp}: {e}")

                if not combined_code.strip():
                    continue

                prompt = f"""
You are a strict senior code reviewer. Review ALL files below and report issues with file path + line numbers.

Return ONLY JSON array like:
[
  {{
    "file": "path/to/file.py",
    "line": 42,
    "type": "bug" | "style" | "performance" | "security" | "best_practice",
    "description": "concise problem explanation",
    "suggestion": "clear fix recommendation"
  }}
]
If no issues, return [].

Code:
{combined_code}
"""

                try:
                    resp = self.groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "Output only valid JSON array."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.15,
                        max_tokens=2000,
                        timeout=120,
                    )

                    text = resp.choices[0].message.content.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                    issues = json.loads(text) if text else []
                    analyzed.append({"batch": i // batch_size + 1, "issues": issues})
                    total_issues += len(issues)
                    logger.info(f"Batch {i // batch_size + 1}: {len(issues)} issues")

                    # Prevent hitting Groq rate limits
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Batch {i // batch_size + 1} failed: {e}")
                    analyzed.append({"batch": i // batch_size + 1, "issues": [], "error": str(e)})

            return {
                "repo_url": repo_url,
                "branch": branch,
                "analyzed_at": datetime.utcnow().isoformat(),
                "total_batches": len(analyzed),
                "total_issues": total_issues,
                "batches": analyzed,
            }

        finally:
            if repo_path and os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)


# -------------------------
# BACKGROUND WORKER
# -------------------------
def process_repo_analysis(task_id: str, repo_url: str, branch: str, github_token: Optional[str] = None):
    try:
        task = get_task(task_id)
        if task:
            task["status"] = "processing"
            save_task(task_id, task)

        logger.info(f"Starting analysis task {task_id}")
        agent = CodebaseReviewAgent(github_token)
        results = agent.analyze_repo(repo_url, branch)

        task = get_task(task_id)
        if task:
            task["status"] = "completed"
            task["results"] = results
            save_task(task_id, task)
        logger.info(f"Completed analysis task {task_id}")

    except Exception as e:
        logger.exception(f"Repo analysis failed for {task_id}: {e}")
        task = get_task(task_id)
        if task:
            task["status"] = "failed"
            task["error"] = str(e)
            save_task(task_id, task)


# -------------------------
# ROUTES
# -------------------------
@app.post("/analyze-repo")
async def analyze_repo(request: Request, repo_request: RepoRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_data = {
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "repo_url": repo_request.repo_url,
        "branch": repo_request.branch,
    }
    save_task(task_id, task_data)
    background_tasks.add_task(process_repo_analysis, task_id, repo_request.repo_url, repo_request.branch, repo_request.github_token)
    logger.info(f"Created repo analysis task {task_id}")
    return {"task_id": task_id, "status": "pending"}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    t = get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "status": t.get("status", "unknown"),
        "created_at": t.get("created_at"),
    }


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    t = get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    status_val = t.get("status")
    if status_val in ("pending", "processing"):
        raise HTTPException(status_code=202, detail=f"Task {status_val}")
    if status_val == "failed":
        raise HTTPException(status_code=500, detail=f"Task failed: {t.get('error')}")
    return {"task_id": task_id, "results": t.get("results", {})}


@app.get("/")
async def root():
    return {
        "message": "AI Codebase Review Agent",
        "version": "2.0.0",
        "storage": "Redis" if redis_client else "In-Memory",
        "endpoints": {
            "POST /analyze-repo": "Start full repo review",
            "GET /status/{task_id}": "Check task status",
            "GET /results/{task_id}": "Get analysis results",
        },
    }


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
