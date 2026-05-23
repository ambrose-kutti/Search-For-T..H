import sys
import os
import logging
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import chromadb
import fitz        # PyMuPDF  — for reading PDF text in viewer
import docx        # python-docx — for reading DOCX in viewer

from api.schemas import SearchRequest, SearchResponse
from search.agent import run_search_agent
from config import ALLOWED_ORIGINS, CHROMA_DIR, CHROMA_COLLECTION

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Resume Search Agent",
    description="Semantic search over resumes using ChromaDB + Ollama. Returns sandbox profile links.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # loaded from .env — no wildcard in production
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


#  Routes 

@app.get("/")
def root():
    return FileResponse("frontend/index1.html")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Search resumes using a natural language query.

    - Accepts free text: skills, names, job titles, or any combination
    - Returns ranked sandbox profile links
    - Optional filters: min_experience (years), location (string)

    Example:
        POST /search
        { "query": "Python developer with FastAPI", "min_experience": 2 }
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"Search request: query='{request.query}' filters=experience:{request.min_experience} location:{request.location}")

    start_time = time.time()

    try:
        # Pass optional filters into the agent
        result = run_search_agent(request.query)

        # Apply any filters passed directly via API (override LLM extracted ones)
        if request.min_experience is not None:
            result["filters_applied"]["min_experience"] = request.min_experience
        if request.location:
            result["filters_applied"]["location"] = request.location

        elapsed_ms = round((time.time() - start_time) * 1000)
        logger.info(f"Search completed in {elapsed_ms}ms | {result['total_results']} results")

        return result

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/stats")
def stats():
    """Returns total number of resumes indexed in ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
        return {
            "total_resumes_indexed": collection.count(),
            "collection": CHROMA_COLLECTION
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  RESUME VIEWER HELPERS 

def _get_resume_metadata(resume_id: str) -> dict:
    """Look up a resume's metadata from ChromaDB by its ID."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    result = collection.get(ids=[resume_id], include=["metadatas"])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail=f"Resume '{resume_id}' not found.")
    return result["metadatas"][0]


def _render_docx_as_html(file_path: str, metadata: dict) -> str:
    """
    Convert a DOCX resume to a clean, readable HTML page.
    Preserves paragraph structure and basic formatting.
    """
    try:
        doc = docx.Document(file_path)
        paragraphs_html = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                paragraphs_html += "<br/>"
                continue
            # Basic heading detection — short lines in bold style
            if para.style.name.startswith("Heading") or (len(text) < 60 and para.runs and para.runs[0].bold):
                paragraphs_html += f'<h3 class="section-head">{text}</h3>'
            else:
                paragraphs_html += f"<p>{text}</p>"
    except Exception as e:
        logger.error(f"DOCX render failed: {e}")
        paragraphs_html = "<p>Could not render document content.</p>"

    name     = metadata.get("candidate_name", "Candidate")
    skills   = metadata.get("skills", "")
    exp      = metadata.get("experience_years", 0)
    location = metadata.get("location", "")
    res_id   = metadata.get("resume_id", "")

    skill_tags = "".join(
        f'<span class="tag">{s.strip()}</span>'
        for s in skills.split(",") if s.strip()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{name} — Resume</title>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0b0f1a;
      color: #e2e8f0;
      font-family: 'DM Sans', sans-serif;
      font-size: 15px;
      line-height: 1.7;
      padding: 0;
    }}
    .top-bar {{
      background: #111827;
      border-bottom: 1px solid #1f2d45;
      padding: 14px 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .top-bar-brand {{
      font-family: 'Syne', sans-serif;
      font-size: 13px;
      font-weight: 700;
      color: #3b82f6;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .top-bar-id {{
      font-size: 12px;
      color: #64748b;
    }}
    .page {{
      max-width: 820px;
      margin: 40px auto;
      padding: 0 24px 80px;
    }}
    .profile-header {{
      background: #161d2e;
      border: 1px solid #1f2d45;
      border-radius: 16px;
      padding: 32px 36px;
      margin-bottom: 28px;
    }}
    .candidate-name {{
      font-family: 'Syne', sans-serif;
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #e2e8f0;
      margin-bottom: 12px;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
      margin-bottom: 20px;
    }}
    .meta-item {{
      font-size: 13px;
      color: #64748b;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .meta-item strong {{ color: #94a3b8; }}
    .tags-wrap {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: 8px;
    }}
    .tag {{
      background: rgba(59,130,246,0.08);
      border: 1px solid rgba(59,130,246,0.18);
      border-radius: 6px;
      color: #93c5fd;
      font-size: 12px;
      font-weight: 500;
      padding: 3px 10px;
    }}
    .resume-body {{
      background: #161d2e;
      border: 1px solid #1f2d45;
      border-radius: 16px;
      padding: 36px 40px;
    }}
    .resume-body p {{
      color: #cbd5e1;
      margin-bottom: 10px;
    }}
    .resume-body br {{ display: block; margin: 6px 0; content: ""; }}
    .section-head {{
      font-family: 'Syne', sans-serif;
      font-size: 15px;
      font-weight: 700;
      color: #3b82f6;
      border-bottom: 1px solid #1f2d45;
      padding-bottom: 6px;
      margin: 24px 0 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
  </style>
</head>
<body>
  <div class="top-bar">
    <span class="top-bar-brand">⚡ Resume Viewer</span>
    <span class="top-bar-id">{res_id}</span>
  </div>
  <div class="page">
    <div class="profile-header">
      <div class="candidate-name">{name}</div>
      <div class="meta-row">
        <span class="meta-item">💼 <strong>{exp}</strong> years experience</span>
        {'<span class="meta-item">📍 <strong>' + location + '</strong></span>' if location and location != "Unknown" else ""}
      </div>
      <div class="tags-wrap">{skill_tags}</div>
    </div>
    <div class="resume-body">
      {paragraphs_html}
    </div>
  </div>
</body>
</html>"""

#  RESUME VIEWER ENDPOINT 

@app.get("/view/{resume_id}")
def view_resume(resume_id: str):
    """
    Resume viewer endpoint.

    - PDF  → served inline so the browser renders it with its built-in PDF viewer
    - DOCX → rendered as a clean HTML page matching the app design

    This is the URL stored as sandbox_link in every search result.
    """
    metadata  = _get_resume_metadata(resume_id)
    file_path = metadata.get("file_path", "")
    file_ext  = metadata.get("file_ext", "")
    file_name = metadata.get("file_name", resume_id)

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Resume file not found on disk: {file_path}. "
                   f"Make sure the resumes folder is intact."
        )

    if file_ext == ".pdf":
        # Return PDF directly — browser opens its built-in PDF viewer
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=file_name,
            headers={"Content-Disposition": "inline"}   # inline = open in browser, not download
        )

    elif file_ext == ".docx":
        # Render DOCX as styled HTML page
        html = _render_docx_as_html(file_path, metadata)
        return HTMLResponse(content=html, status_code=200)

    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file_ext}'. Only PDF and DOCX are supported."
        )