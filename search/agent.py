"""
search/agent.py — LangChain-style search agent using Ollama.

Architecture:
  1. LLM extracts structured filters from the query
     (tech_skills list, min_experience, location)
  2. Original user query is embedded directly — no LLM rewriting
     (preserves exact technical terms for best vector match)
  3. ChromaDB applies strict filters + vector search
  4. Results re-ranked by skills match score
"""

import sys
import os
import json
import re
import logging
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, LLM_MODEL
from search.chroma_search import search_resumes
logger = logging.getLogger(__name__)

def _call_llm(prompt: str) -> str:
    """Send a prompt to Ollama and return response text."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Make sure it is running.")
        return ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

def extract_search_intent(user_query: str) -> dict:
    """
    Use LLM to extract structured filters from the user's query.

    The LLM does NOT rewrite the query — it only extracts:
      - tech_skills: exact technology names mentioned
      - min_experience: years if explicitly mentioned
      - location: place if explicitly mentioned
      - explanation: human-readable summary
    The original query is used as-is for vector embedding to preserve
    exact technical terms (SAP, ABAP, Python, etc.)
    """
    prompt = f"""You are a resume search filter extractor.
                    Extract structured filters from the user's search query.

                    User query: "{user_query}"

                    Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.

                    {{
                    "tech_skills": ["exact skill names from query"],
                    "min_experience": null,
                    "location": null,
                    "explanation": "one sentence summary of what user wants"
                    }}

                    Rules:
                    - tech_skills: extract EXACT technology/tool/language names as the user typed them.
                    Examples: "python ML" → ["Python", "Machine Learning"]
                                "SAP ABAP"  → ["SAP", "ABAP"]
                                "react developer" → ["React"]
                                "java spring boot" → ["Java", "Spring Boot"]
                    - min_experience: integer ONLY if the user explicitly says years (e.g. "3 years" → 3). Otherwise null.
                    - location: string ONLY if the user explicitly mentions a place. Otherwise null.
                    - Do NOT invent skills not mentioned. Do NOT rewrite or expand the query.
"""
    raw = _call_llm(prompt)
    if not raw:
        return {
            "tech_skills":    [],
            "min_experience": None,
            "location":       None,
            "explanation":    user_query
        }
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"```$", "", clean).strip()
        # Find JSON object in response
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object in response")
        intent = json.loads(clean[start:end])
        # Ensure tech_skills is always a list
        if not isinstance(intent.get("tech_skills"), list):
            intent["tech_skills"] = []
        logger.info(f"Extracted intent: {intent}")
        return intent
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Intent parse failed: {e}. Falling back to raw query.")
        return {
            "tech_skills":    [],
            "min_experience": None,
            "location":       None,
            "explanation":    user_query
        }

def run_search_agent(user_query: str) -> dict:
    """
    Full agent pipeline:
    Step 1 — LLM extracts filters (tech_skills, min_experience, location)
    Step 2 — Embed ORIGINAL query (not rewritten) for faithful vector search
    Step 3 — ChromaDB: strict filters + vector search + skills re-ranking
    Step 4 — Return structured response with sandbox links
    Args:
        user_query: Raw natural language query from the user
    Returns:
        Structured dict with results, filters applied, and explanation
    """
    logger.info(f"Agent received query: '{user_query}'")

    # Step 1: Extract filters via LLM
    intent = extract_search_intent(user_query)
    tech_skills    = intent.get("tech_skills", [])
    min_experience = intent.get("min_experience")
    location       = intent.get("location")
    explanation    = intent.get("explanation", user_query)

    # Step 2 & 3: Search — original query embedded, filters applied strictly
    filters = {}
    if min_experience is not None:
        filters["min_experience"] = int(min_experience)
    if location:
        filters["location"] = location
    results = search_resumes(
        query      = user_query,      # original query — not rewritten
        filters    = filters,
        tech_skills= tech_skills      # for skills-aware re-ranking
    )
    return {
        "original_query":  user_query,
        "understood_as":   explanation,
        "search_query":    user_query,
        "filters_applied": filters,
        "tech_skills":     tech_skills,
        "total_results":   len(results),
        "results":         results
    }