"""
nlp_extractor.py — Production-grade metadata extraction for resumes.

Strategy:
  - Candidate name   → filename parse first (Naukri format), spaCy NER as fallback
  - Experience years → filename parse first ([Xy_Ym] format), regex on text as fallback
  - Location         → spaCy NER
  - Skills           → Ollama LLM (no hardcoded list)
"""
import re
import sys
import os
import json
import logging
import requests
import spacy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, LLM_MODEL
logger = logging.getLogger(__name__)

# spaCy 
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    _nlp = None

# Experience regex (resume text) 
_EXP_PATTERNS = [
    r"(\d+)\+?\s*years?\s+of\s+experience",
    r"(\d+)\+?\s*years?\s+experience",
    r"experience\s+of\s+(\d+)\+?\s*years?",
    r"(\d+)\+?\s*yrs?\s+of\s+experience",
    r"(\d+)\+?\s*yrs?\s+experience",
]

# FILENAME-BASED EXTRACTORS
def _split_camel_case(text: str) -> str:
    """
    Split CamelCase or PascalCase into separate words.
    e.g. "SubhasishSingha" → "Subhasish Singha"
         "KoushalRathor"   → "Koushal Rathor"
    """
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text).strip()

def extract_name_from_filename(file_name: str) -> str:
    """
    Extract candidate name from resume filename.

    Handles patterns:
      - Naukri_FirstnameLastname[Xy_Ym].pdf  → "Firstname Lastname"
      - Naukri_FirstnameLastname.pdf          → "Firstname Lastname"
      - John_Doe_Resume.pdf                   → "John Doe"
      - JohnDoe.pdf                           → "John Doe"

    Returns empty string if no name can be parsed confidently.
    """
    base = os.path.splitext(file_name)[0]   # Strip extension
    base = re.sub(r"\[\d+y_\d+m\]", "", base).strip()   # Remove experience bracket [12y_0m] if present
    base = re.sub(r"(?i)(resume|cv|profile|updated|new|final|naukri)_?", "", base).strip("_- ") # Remove common suffixes
    parts = re.split(r"[_\-]", base)    # Split by underscore or hyphen, take remaining parts
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""

    # Rejoin and split CamelCase
    joined = " ".join(parts)
    name   = _split_camel_case(joined)

    # Basic sanity check — must look like a name (2+ chars, not all digits)
    if len(name) < 3 or name.isdigit():
        return ""
    return name.title()

def extract_experience_from_filename(file_name: str) -> int:
    """
    Extract years of experience from Naukri-style filename bracket.

    Pattern: [12y_0m] → 12 years
             [9y_6m]  → 9 years
             [0y_8m]  → 0 years

    Returns -1 if no bracket pattern found (so caller knows to fall back).
    """
    match = re.search(r"\[(\d+)y_(\d+)m\]", file_name)
    if match:
        return int(match.group(1))
    return -1

#  TEXT-BASED EXTRACTORS (fallbacks)
def extract_name_from_text(text: str) -> str:
    """Extract name using spaCy NER — used as fallback if filename parse fails."""
    if not _nlp:
        return "Unknown"
    doc = _nlp(text[:600])
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
            return ent.text.strip()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return "Unknown"

def extract_experience_from_text(text: str) -> int:
    """Extract years of experience from resume text using regex."""
    found = []
    text_lower = text.lower()
    for pattern in _EXP_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            found.append(int(match.group(1)))
    return max(found) if found else 0

def extract_location(text: str) -> str:
    """Extract location using spaCy GPE."""
    if not _nlp:
        return "Unknown"
    doc = _nlp(text[:1500])
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            return ent.text.strip()
    return "Unknown"

def extract_skills_via_llm(text: str) -> list[str]:
    """
    Use Ollama LLM to extract all technical skills from resume text.
    No hardcoded skills list — LLM reads the resume and identifies everything.
    Falls back to empty list if LLM is unavailable.
    """
    excerpt = text[:3000].strip()
    prompt = f"""You are a resume parser. Extract ALL technical skills from the resume text below.

                Include: programming languages, frameworks, libraries, tools, platforms, databases,
                cloud services, methodologies, and any other technical competencies mentioned.

                Resume text:
                \"\"\"
                {excerpt}
                \"\"\"

                Rules:
                - Respond ONLY with a valid JSON array of strings.
                - Each item must be a single skill or technology name.
                - Normalise capitalisation: "Python" not "python", "AWS" not "aws", "SAP ABAP" not "sap abap".
                - Do NOT include soft skills (communication, leadership, teamwork etc.).
                - Do NOT include job titles or company names.
                - Do NOT add any explanation, markdown, or text outside the JSON array.

                Example output format:
                ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "React"]
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=90
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"```$", "", clean).strip()
        # Find JSON array anywhere in response — handles LLM preamble text
        start = clean.find("[")
        end   = clean.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON array found in LLM response")
        clean = clean[start : end + 1]
        skills = json.loads(clean)
        if not isinstance(skills, list):
            raise ValueError("LLM response is not a JSON array")
        skills = list({s.strip() for s in skills if isinstance(s, str) and s.strip()})
        logger.info(f"LLM extracted {len(skills)} skills")
        return skills
    except requests.exceptions.ConnectionError:
        logger.error("Ollama not reachable during skill extraction.")
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Skill extraction parse failed: {e}. Raw: {raw[:200]}")
        return []
    except Exception as e:
        logger.error(f"Skill extraction failed: {e}")
        return []

# MAIN ENTRY POINT
def extract_metadata(text: str, file_name: str = "") -> dict:
    """
    Extract all metadata from resume text + filename.

    Priority order:
      name       → filename parse → spaCy NER → "Unknown"
      experience → filename bracket [Xy_Ym] → regex on text → 0
      location   → spaCy NER → "Unknown"
      skills     → Ollama LLM → []
    """
    #  Name 
    name = ""
    if file_name:
        name = extract_name_from_filename(file_name)
    if not name or name == "Unknown":
        name = extract_name_from_text(text)
    #  Experience 
    exp = -1
    if file_name:
        exp = extract_experience_from_filename(file_name)
    if exp == -1:
        exp = extract_experience_from_text(text)
    #  Skills 
    skills = extract_skills_via_llm(text)
    metadata = {
        "candidate_name":   name or "Unknown",
        "skills":           ", ".join(skills),
        "experience_years": exp if exp >= 0 else 0,
        "location":         extract_location(text),
        "skills_count":     len(skills),
    }
    logger.debug(f"Metadata for {file_name}: {metadata}")
    return metadata