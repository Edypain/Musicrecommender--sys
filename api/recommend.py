from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import os
import joblib
import requests
import time

app = FastAPI(
    title="Music Recommender API",
    description="A hybrid music recommendation system using content-based and collaborative filtering",
    version="1.0.0"
)

# runtime flags
DATA_LOADED = False
load_error_message = None

# data containers
similarity_matrix = None
song_db = None
svd = None
valid_files = []
moods = []
clusters = []

# --- Pydantic models ---
class RecommendationRequest(BaseModel):
    song_index: Optional[int] = None
    song_name: Optional[str] = None
    user_id: int
    num_recommendations: int = 5
    same_cluster: bool = True
    mood_filter: Optional[str] = None

class SongRecommendation(BaseModel):
    song_name: str
    mood: Optional[str]
    cluster: Optional[int]
    score: float

class RecommendationResponse(BaseModel):
    seed_song: str
    recommendations: List[SongRecommendation]

class HealthResponse(BaseModel):
    status: str
    total_songs: int
    loaded: bool

# --- Helpers ---

def safe_load_models():
    """Attempt to load models and datasets. If files are missing, set DATA_LOADED=False
    and keep a helpful error message."""
    global similarity_matrix, song_db, svd, valid_files, moods, clusters, DATA_LOADED, load_error_message
    try:
        base = os.path.dirname(os.path.dirname(__file__)) or '.'
        models_dir = os.path.join(base, 'models')
        svd_path = os.path.join(models_dir, 'svd.pkl')
        sim_path = os.path.join(models_dir, 'similarity_matrix.npy')
        dataset_path = os.path.join(base, 'music_dataset.csv')

        # Require similarity matrix and dataset. svd is optional (collaborative fallback).
        missing = []
        if not os.path.exists(sim_path):
            missing.append(('models/similarity_matrix.npy', sim_path))
        if not os.path.exists(dataset_path):
            missing.append(('music_dataset.csv', dataset_path))

        # If files are missing, attempt to download from MODEL_BASE_URL if provided.
        model_base = os.environ.get('MODEL_BASE_URL')
        if missing and model_base:
            os.makedirs(models_dir, exist_ok=True)
            downloaded = []
            for rel, abs_path in missing:
                url = model_base.rstrip('/') + '/' + rel.lstrip('/')
                # try download with a few retries
                ok = False
                for attempt in range(3):
                    try:
                        resp = requests.get(url, timeout=10)
                        if resp.status_code == 200:
                            with open(abs_path, 'wb') as fh:
                                fh.write(resp.content)
                            downloaded.append(abs_path)
                            ok = True
                            break
                        else:
                            print(f"Download attempt {attempt+1} for {url} returned status {resp.status_code}")
                    except Exception as e:
                        print(f"Download attempt {attempt+1} for {url} failed: {e}")
                    time.sleep(1 + attempt)
                if not ok:
                    print(f"Failed to download {url}")

            # recompute missing list after attempts
            missing = [item for item in missing if not os.path.exists(item[1])]

        if missing:
            # Format missing for message
            load_error_message = f"Missing required files: {', '.join([p for p,_ in missing])}"
            DATA_LOADED = False
            return

        # Load mandatory artifacts
        similarity_matrix = np.load(sim_path)
        song_db = pd.read_csv(dataset_path)

        valid_files = song_db['file'].tolist()
        moods = song_db['mood'].tolist() if 'mood' in song_db.columns else [None] * len(valid_files)
        clusters = song_db['cluster'].tolist() if 'cluster' in song_db.columns else [None] * len(valid_files)

        # Try to load optional svd model; if it fails, continue without it.
        if os.path.exists(svd_path):
            try:
                svd = joblib.load(svd_path)
            except Exception as e:
                svd = None
                print(f"⚠️ Warning: failed to load svd model ({svd_path}): {e}. Continuing without collaborative filtering.")

        DATA_LOADED = True
        load_error_message = None
        print("✅ Preprocessed data loaded successfully (SVD present: {} )".format('yes' if svd is not None else 'no'))
    except Exception as e:
        DATA_LOADED = False
        load_error_message = str(e)
        print(f"❌ Error loading preprocessed data: {e}")


@app.on_event("startup")
async def startup_event():
    safe_load_models()


def find_song_index(song_name: str) -> int:
    """Find song index by name (case-insensitive partial match)"""
    song_name_lower = song_name.lower()
    for i, file_name in enumerate(valid_files):
        if song_name_lower in file_name.lower():
            return i
    raise HTTPException(status_code=404, detail=f"Song '{song_name}' not found")


def get_collaborative_recommendations(user_id, num_recommendations=5):
    """Get collaborative filtering recommendations. Returns list of song indices."""
    if svd is None:
        return []
    predictions = []
    for song_id in range(len(valid_files)):
        try:
            pred = svd.predict(user_id, song_id)
            predictions.append((song_id, pred.est))
        except Exception:
            # If svd.predict fails for some pair, skip
            continue
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in predictions[:num_recommendations]]


@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if DATA_LOADED else "degraded",
        total_songs=len(valid_files) if valid_files else 0,
        loaded=DATA_LOADED
    )


@app.get("/songs")
async def get_all_songs():
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail=load_error_message or "Data not loaded")
    return {
        "songs": [
            {"index": i, "name": song, "mood": (moods[i] if i < len(moods) else None), "cluster": (int(clusters[i]) if i < len(clusters) and clusters[i] is not None else None)}
            for i, song in enumerate(valid_files)
        ]
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_songs(request: RecommendationRequest):
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail=load_error_message or "Data not loaded")

    # Find the query index
    if request.song_index is not None:
        if request.song_index >= len(valid_files):
            raise HTTPException(status_code=400, detail="Invalid song index")
        query_index = request.song_index
    elif request.song_name is not None:
        query_index = find_song_index(request.song_name)
    else:
        raise HTTPException(status_code=400, detail="Either song_index or song_name must be provided")

    # Basic user_id validation (make it permissive)
    if request.user_id < 0:
        raise HTTPException(status_code=400, detail="User ID must be non-negative")

    # Content-based scores
    try:
        content_scores = list(enumerate(similarity_matrix[query_index]))
    except Exception:
        raise HTTPException(status_code=500, detail="Similarity matrix not usable")

    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)

    # Collaborative
    collab_recs = get_collaborative_recommendations(request.user_id, request.num_recommendations * 2)

    hybrid_weight = 0.5
    query_cluster = clusters[query_index] if query_index < len(clusters) else None
    query_mood = moods[query_index] if query_index < len(moods) else None

    hybrid_candidates = {}
    for idx, content_score in content_scores:
        if idx == query_index:
            continue
        if request.same_cluster and query_cluster is not None and (idx >= len(clusters) or clusters[idx] != query_cluster):
            continue
        if request.mood_filter and (idx >= len(moods) or moods[idx] != request.mood_filter):
            continue

        hybrid_score = (hybrid_weight * content_score +
                        (1 - hybrid_weight) * (5 if idx in collab_recs else 0))
        hybrid_candidates[idx] = hybrid_score

    top_indices = sorted(hybrid_candidates, key=hybrid_candidates.get, reverse=True)[:request.num_recommendations]

    recommendations = []
    for idx in top_indices:
        recommendations.append(SongRecommendation(
            song_name=valid_files[idx],
            mood=(moods[idx] if idx < len(moods) else None),
            cluster=(int(clusters[idx]) if idx < len(clusters) and clusters[idx] is not None else None),
            score=float(hybrid_candidates[idx])
        ))

    return RecommendationResponse(
        seed_song=valid_files[query_index],
        recommendations=recommendations
    )


@app.get("/recommend/random")
async def recommend_random_songs(user_id: int = 1, num_recommendations: int = 5):
    import random
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail=load_error_message or "Data not loaded")
    random_index = random.randint(0, len(valid_files) - 1)

    request = RecommendationRequest(
        song_index=random_index,
        user_id=user_id,
        num_recommendations=num_recommendations
    )

    return await recommend_songs(request)


@app.get("/song/{song_index}")
async def get_song_details(song_index: int):
    if not DATA_LOADED:
        raise HTTPException(status_code=503, detail=load_error_message or "Data not loaded")
    if song_index >= len(valid_files):
        raise HTTPException(status_code=404, detail="Song index not found")

    # Safely extract features if they exist
    row = song_db.iloc[song_index]
    features = {}
    for f in ['tempo', 'rms', 'centroid']:
        if f in song_db.columns:
            features[f] = float(row[f])

    return {
        "index": song_index,
        "name": valid_files[song_index],
        "mood": moods[song_index] if song_index < len(moods) else None,
        "cluster": int(clusters[song_index]) if song_index < len(clusters) and clusters[song_index] is not None else None,
        "features": features
    }


# NOTE: We rely on the FastAPI startup event to load models (recommended for production).
# Tests should create TestClient inside their test functions so startup events run.
