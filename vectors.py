import json
import os
import numpy as np
from utils import get_embeddings

# Path to store strong resumes
STRONG_RESUMES_FILE = "strong_resumes.json"

# In-memory storage for strong resumes
strong_resumes_db = []

def load_strong_resumes():
    global strong_resumes_db
    if os.path.exists(STRONG_RESUMES_FILE):
        try:
            with open(STRONG_RESUMES_FILE, 'r') as f:
                strong_resumes_db = json.load(f)
        except Exception:
            strong_resumes_db = []
    else:
        strong_resumes_db = []

def save_strong_resumes():
    #save strong resumes to JSON file
    try:
        with open(STRONG_RESUMES_FILE, 'w') as f:
            json.dump(strong_resumes_db, f, indent=2)
    except Exception:
        pass

def add_strong_resume(candidate_data):
    #Add a strong resume to the database
    load_strong_resumes()  # Ensure loaded
    # Generate embedding if not present
    if 'embedding' not in candidate_data:
        resume_text = candidate_data.get('resume_text', '')
        if resume_text:
            candidate_data['embedding'] = get_embeddings(resume_text)
    strong_resumes_db.append(candidate_data)
    save_strong_resumes()

def find_similar_resumes(query_embedding, top_k=3):
    """Find similar resumes based on embedding similarity using manual cosine similarity."""
    load_strong_resumes()
    if not query_embedding:
        return []

    # Manual cosine similarity
    similarities = []
    for resume in strong_resumes_db:
        emb = resume.get('embedding')
        if emb:
            # Simple cosine similarity (assuming normalized embeddings)
            sim = sum(a*b for a,b in zip(query_embedding, emb))
            similarities.append({
                'candidate_id': resume.get('candidate_id', 'unknown'),
                'score': round(sim * 100, 2),
                'metadata': resume
            })
    similarities.sort(key=lambda x: x['score'], reverse=True)
    return similarities[:top_k]
# Load on import
load_strong_resumes()
