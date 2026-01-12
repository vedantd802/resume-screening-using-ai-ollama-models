import json
import os
import math

# Use dynamic import from utils to avoid hard dependency on numpy
from utils import _try_import
np = _try_import('numpy')
_NUMPY_AVAILABLE = np is not None

STORE_FILE = "strong_resumes.json"

def load_store():
    if not os.path.exists(STORE_FILE):
        return []
    with open(STORE_FILE, "r") as f:
        return json.load(f)

def save_store(data):
    with open(STORE_FILE, "w") as f:
        json.dump(data, f)

def add_resume_embedding(candidate_id, embedding, metadata=None):
    """
    Stores strong resume embedding with candidate ID and metadata (e.g., phone, email, etc.)
    """
    data = load_store()

    record = {
        "candidate_id": candidate_id,
        "embedding": embedding,
        "metadata": metadata or {}
    }

    data.append(record)
    save_store(data)

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def _norm(a):
    return math.sqrt(sum(x * x for x in a))

def find_similar_resumes(query_embedding, top_k=5):
    """
    Finds top matching stored resumes.
    Returns candidate_id, score, and metadata for each match.
    """
    data = load_store()
    if not data:
        return []

    scores = []
    if _NUMPY_AVAILABLE:
        query_arr = np.array(query_embedding)
    else:
        query_arr = list(query_embedding)

    for item in data:
        emb = item.get("embedding")
        if emb is None:
            continue

        try:
            if _NUMPY_AVAILABLE:
                emb_arr = np.array(emb)
                similarity = float(np.dot(query_arr, emb_arr) / (np.linalg.norm(query_arr) * np.linalg.norm(emb_arr)))
            else:
                emb_arr = list(emb)
                denom = _norm(query_arr) * _norm(emb_arr)
                similarity = _dot(query_arr, emb_arr) / denom if denom != 0 else 0.0
        except Exception:
            similarity = 0.0

        scores.append({
            "candidate_id": item["candidate_id"],
            "score": round(similarity * 100, 2),
            "metadata": item.get("metadata", {})  # <-- Return metadata
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]
