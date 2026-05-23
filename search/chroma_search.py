"""
search/chroma_search.py — ChromaDB query with strict filtering and skills re-ranking.
Search strategy:
  1. Apply strict metadata filters (experience >= N, location contains X)
  2. Vector similarity search on filtered pool
  3. Re-rank by skills match score
  4. Exclude results with zero skills match when tech skills are specified
Combined score = (0.55 × vector_similarity) + (0.45 × skills_match_ratio)
"""

import sys
import os
import logging
import chromadb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHROMA_DIR, CHROMA_COLLECTION,
    OLLAMA_BASE_URL, EMBEDDING_MODEL, TOP_K_RESULTS
)
from ingestion.embedder import get_embedding
logger = logging.getLogger(__name__)

def get_collection():
    """Connect to ChromaDB and return the resumes collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

def build_where_filter(filters: dict) -> dict | None:
    """
    Build strict ChromaDB where= filter.
    min_experience → experience_years >= value  (hard constraint, not suggestion)
    location       → location $contains value   (partial match)
    """
    conditions = []
    if filters.get("min_experience") is not None:
        exp = int(filters["min_experience"])
        conditions.append({"experience_years": {"$gte": exp}})
        logger.info(f"Strict filter: experience_years >= {exp}")
    if filters.get("location"):
        loc = filters["location"].strip()
        conditions.append({"location": {"$contains": loc}})
        logger.info(f"Strict filter: location contains '{loc}'")
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def skills_match_score(resume_skills_str: str, tech_skills: list) -> float:
    """
    Calculate what fraction of queried tech skills appear in the resume.
    Returns 0.0–1.0. 1.0 = every queried skill found in resume.
    Case-insensitive substring match.
    """
    if not tech_skills:
        return 0.0
    resume_lower = resume_skills_str.lower()
    matched = sum(1 for skill in tech_skills if skill.lower() in resume_lower)
    return matched / len(tech_skills)

def search_resumes(
    query:       str,
    filters:     dict = {},
    tech_skills: list = []
) -> list[dict]:
    """
    Search resumes with strict filters and skills-aware re-ranking.
    Args:
        query:       Original user query (embedded as-is for best vector match)
        filters:     min_experience (int), location (str) — applied as hard constraints
        tech_skills: Tech keywords from query — used for skills match scoring
    Returns:
        Ranked list of results. Resumes with zero skills match are excluded
        when tech_skills are specified.
    """
    collection = get_collection()
    total = collection.count()
    if total == 0:
        logger.warning("ChromaDB collection is empty. Run ingestion first.")
        return []

    # Step 1: Embed original query
    logger.info(f"Embedding query: '{query}' | tech_skills={tech_skills} | filters={filters}")
    query_embedding = get_embedding(query, OLLAMA_BASE_URL, EMBEDDING_MODEL)
    if not query_embedding:
        logger.error("Failed to embed query. Is Ollama running?")
        return []

    # Step 2: Strict metadata filter
    where_filter = build_where_filter(filters)

    # Fetch 3x results to have a pool for re-ranking
    fetch_n = min(TOP_K_RESULTS * 3, total)
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results":        fetch_n,
        "include":          ["metadatas", "distances", "documents"]
    }
    if where_filter:
        query_params["where"] = where_filter
    try:
        raw = collection.query(**query_params)
    except Exception as e:
        logger.error(f"ChromaDB query error: {e}")
        return []
    if not raw["metadatas"] or not raw["metadatas"][0]:
        logger.info("No results matched the filters.")
        return []

    # Step 3: Score and re-rank
    candidates = []
    for i, metadata in enumerate(raw["metadatas"][0]):
        distance     = raw["distances"][0][i]
        vector_score = round(1 - distance, 4)
        s_score      = skills_match_score(metadata.get("skills", ""), tech_skills)
        # Combined score
        if tech_skills:
            combined = round((0.55 * vector_score) + (0.45 * s_score), 4)
        else:
            combined = vector_score
        # Exclude zero-skills-match when tech skills were specified
        if tech_skills and s_score == 0.0:
            logger.debug(f"Excluded (no skills match): {metadata.get('file_name')}")
            continue
        candidates.append({
            "vector_score":   vector_score,
            "skills_score":   round(s_score, 4),
            "combined_score": combined,
            "metadata":       metadata,
        })
    # Sort by combined score
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    top = candidates[:TOP_K_RESULTS]
    # Step 4: Format output
    results = []
    for rank, c in enumerate(top, start=1):
        meta   = c["metadata"]
        skills = [s.strip() for s in meta.get("skills", "").split(",") if s.strip()]
        results.append({
            "rank":             rank,
            "similarity_score": c["combined_score"],
            "vector_score":     c["vector_score"],
            "skills_score":     c["skills_score"],
            "sandbox_link":     meta.get("sandbox_url", "N/A"),
            "candidate_name":   meta.get("candidate_name", "Unknown"),
            "matched_skills":   skills,
            "experience_years": meta.get("experience_years", 0),
            "location":         meta.get("location", "Unknown"),
            "file_name":        meta.get("file_name", ""),
        })
    logger.info(f"Returned {len(results)} results for '{query}'")
    return results