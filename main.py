import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from github import Github
import logging
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in .env file or export it.")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set. Please set it in .env file or export it.")

app = FastAPI(title="Codebase Analysis Agent", version="1.0.0")

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://code-bot-rho.vercel.app",
    "https://*.vercel.app",  # Allow all Vercel preview deployments
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
github_client = Github(GITHUB_TOKEN)

class AnalysisRequest(BaseModel):
    repository: str
    type: str = "general"

class AnalysisResponse(BaseModel):
    success: bool
    repository: str
    analysis_type: str
    analysis: str
    files_analyzed: int
    stars: int
    language: Optional[str]

def parse_repository_name(repo_input: str) -> str:
    """Parse repository name from various input formats"""
    # Remove trailing slashes
    repo_input = repo_input.rstrip('/')
    
    # If it's a URL, extract owner/repo
    if 'github.com' in repo_input:
        parts = repo_input.split('github.com/')
        if len(parts) > 1:
            return parts[1]
    
    # If it already looks like owner/repo, return as is
    if '/' in repo_input and not repo_input.startswith('http'):
        return repo_input
    
    # Otherwise return as is and let it fail with a clear error
    return repo_input

def get_repository_structure(repo, path="", max_depth=3, current_depth=0) -> List[str]:
    """Recursively get repository structure"""
    if current_depth >= max_depth:
        return []
    
    structure = []
    try:
        contents = repo.get_contents(path)
        if not isinstance(contents, list):
            contents = [contents]
        
        for content in contents:
            if content.type == "dir":
                structure.append(f"📁 {content.path}/")
                structure.extend(get_repository_structure(repo, content.path, max_depth, current_depth + 1))
            else:
                size_kb = content.size / 1024
                structure.append(f"📄 {content.path} ({size_kb:.1f}KB)")
    except Exception as e:
        logger.error(f"Error reading {path}: {str(e)}")
    
    return structure

def get_key_files(repo, extensions=['.py', '.js', '.ts', '.java', '.go', '.rs', '.jsx', '.tsx', '.vue', '.cpp', '.c', '.rb']):
    """Get content of key files from repository, recursively exploring all directories"""
    key_files = []
    files_to_process = []
    
    try:
        # Get all contents recursively
        contents = repo.get_contents("")
        
        while contents:
            file_content = contents.pop(0)
            
            if file_content.type == "dir":
                # Explore subdirectories (including app/, src/, components/, etc.)
                try:
                    contents.extend(repo.get_contents(file_content.path))
                except Exception as e:
                    logger.warning(f"Could not access directory {file_content.path}: {str(e)}")
            else:
                # Check if file has relevant extension
                if any(file_content.path.endswith(ext) for ext in extensions):
                    if file_content.size < 50000:  # Reduced to 50KB max per file
                        files_to_process.append(file_content)
        
        # Prioritize important files
        priority_patterns = [
            'main.', 'app.', 'index.', 'server.', 'api.',
            'config.', 'route', 'controller', 'model',
            'service', 'component', 'page', 'layout'
        ]
        
        # Sort files by priority
        def file_priority(file):
            path_lower = file.path.lower()
            for i, pattern in enumerate(priority_patterns):
                if pattern in path_lower:
                    return i
            return len(priority_patterns)
        
        files_to_process.sort(key=file_priority)
        
        # Process files up to limit
        for file_content in files_to_process[:10]:  # Reduced to 10 files
            try:
                content = file_content.decoded_content.decode('utf-8')
                key_files.append({
                    'path': file_content.path,
                    'content': content
                })
                logger.info(f"Analyzed file: {file_content.path}")
            except Exception as e:
                logger.warning(f"Could not read file {file_content.path}: {str(e)}")
                
            if len(key_files) >= 10:
                break
    
    except Exception as e:
        logger.error(f"Error getting key files: {str(e)}")
    
    return key_files

def analyze_codebase(repo_full_name: str, analysis_type: str = "general") -> dict:
    """Analyze codebase using Groq"""
    try:
        # Parse repository name if URL is provided
        repo_full_name = parse_repository_name(repo_full_name)
        repo = github_client.get_repo(repo_full_name)
        
        # Get repository structure
        structure = get_repository_structure(repo)
        structure_text = "\n".join(structure[:50])  # Reduced to 50 items
        
        # Get key files
        key_files = get_key_files(repo)
        files_text = "\n\n".join([
            f"=== {f['path']} ===\n{f['content'][:1000]}"  # Reduced to 1000 chars per file
            for f in key_files[:10]  # Reduced to 10 files max
        ])
        
        # Get README
        readme_content = ""
        try:
            readme = repo.get_readme()
            readme_content = readme.decoded_content.decode('utf-8')[:1000]  # Reduced to 1000 chars
        except:
            pass
        
        # Prepare prompt based on analysis type
        prompts = {
            "general": "Provide a comprehensive analysis of this codebase including: architecture overview, tech stack, code quality, potential improvements, and security considerations.",
            "security": "Perform a security analysis focusing on: potential vulnerabilities, security best practices, authentication/authorization issues, and data handling concerns.",
            "architecture": "Analyze the architecture focusing on: design patterns, code organization, modularity, scalability, and architectural improvements.",
            "quality": "Assess code quality focusing on: code smells, best practices adherence, maintainability, testing coverage, and refactoring suggestions.Rate the code quality out of 100 at the top.",
            "performance": "Analyze performance aspects including: potential bottlenecks, optimization opportunities, resource usage, and scalability concerns."
        }
        
        prompt = f"""Analyze this codebase:

Repository: {repo_full_name}
Description: {repo.description or 'N/A'}

README:
{readme_content}

Repository Structure:
{structure_text}

Key Files:
{files_text}

Task: {prompts.get(analysis_type, prompts['general'])}

Provide a detailed, actionable analysis."""

        # Try multiple models in order of preference (all currently active as of Oct 2025)
        models = [
            ("llama-3.3-70b-versatile", 4000),      # Primary - best quality, reduced tokens
            ("qwen/qwen3-32b", 6000),                # Large context fallback
            ("gemma2-9b-it", 4000),                  # Google's efficient model
            ("llama-3.1-8b-instant", 4000)           # Fast fallback, reduced tokens
        ]
        
        analysis = None
        last_error = None
        
        for model_name, max_tokens in models:
            try:
                # Call Groq API
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model_name,
                    temperature=0.5,
                    max_tokens=max_tokens,
                )
                
                analysis = chat_completion.choices[0].message.content
                logger.info(f"Successfully used model: {model_name}")
                break
                
            except Exception as model_error:
                last_error = str(model_error)
                logger.warning(f"Failed with {model_name}: {last_error}")
                if "rate_limit" not in last_error.lower() and model_name == models[0][0]:
                    # If it's not a rate limit error on first model, don't try others
                    break
                continue
        
        if not analysis:
            return {
                "success": False,
                "error": f"All models failed. Last error: {last_error}"
            }
        
        return {
            "success": True,
            "repository": repo_full_name,
            "analysis_type": analysis_type,
            "analysis": analysis,
            "files_analyzed": len(key_files),
            "stars": repo.stargazers_count,
            "language": repo.language
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Codebase Analysis Agent",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "webhook": "/webhook"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "codebase-analyzer"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Main analysis endpoint"""
    try:
        valid_types = ['general', 'security', 'architecture', 'quality', 'performance']
        if request.type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid analysis type. Valid types: {valid_types}"
            )
        
        result = analyze_codebase(request.repository, request.type)
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Analysis failed'))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook")
async def github_webhook(request: Request):
    """GitHub webhook endpoint for automated analysis"""
    try:
        event = request.headers.get('X-GitHub-Event')
        payload = await request.json()
        
        if event == 'push':
            repo_name = payload['repository']['full_name']
            
            # Trigger analysis on push to main/master
            if payload['ref'] in ['refs/heads/main', 'refs/heads/master']:
                result = analyze_codebase(repo_name, 'general')
                logger.info(f"Automated analysis completed for {repo_name}")
                return result
        
        return {"message": "Event received"}
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)