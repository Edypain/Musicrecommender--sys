from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # for saving/loading models
import uvicorn
# Note: librosa, sklearn preprocessing, PCA, KMeans are no longer needed
# in main.py because we are only LOADING models, not training them.
# We also removed multiprocessing.

app = FastAPI(
    title="Music Recommender API",
    description="A hybrid music recommendation system using content-based and collaborative filtering",
    version="1.0.0"
)

# Global variables to store your models and data
# feature_matrix is no longer needed, as similarity_matrix is pre-calculated
similarity_matrix = None
song_db = None
svd = None
valid_files = None
moods = None
clusters = None

# --- Pydantic models (No changes needed) ---
class RecommendationRequest(BaseModel):
    song_index: Optional[int] = None
    song_name: Optional[str] = None
    user_id: int
    num_recommendations: int = 5
    same_cluster: bool = True
    mood_filter: Optional[str] = None

class SongRecommendation(BaseModel):
    song_name: str
    mood: str
    cluster: int
    score: float

class RecommendationResponse(BaseModel):
    seed_song: str
    recommendations: List[SongRecommendation]

class HealthResponse(BaseModel):
    status: str
    total_songs: int
    loaded: bool

# --- END Pydantic models ---


@app.on_event("startup")
async def startup_event():
    """Initialize the recommender system when the app starts"""
    global similarity_matrix, song_db, svd, valid_files, moods, clusters
    
    try:
        # We ONLY load preprocessed data.
        await load_preprocessed_data()
        print(f"✅ Music Recommender initialized with {len(valid_files)} songs")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        # This will cause the deployment to fail, which is correct
        # if the models are missing.
        raise e

# REMOVED the entire preprocess_data() function.
# It now lives in run_preprocessing.py locally.

async def load_preprocessed_data():
    """Load preprocessed data and models"""
    global similarity_matrix, song_db, svd, valid_files, moods, clusters
    
    try:
        # We don't need scaler, pca, or kmeans anymore for recommendations,
        # only for preprocessing.
        svd = joblib.load('models/svd.pkl')
        similarity_matrix = np.load('models/similarity_matrix.npy')
        song_db = pd.read_csv('music_dataset.csv')
        
        valid_files = song_db['file'].tolist()
        moods = song_db['mood'].tolist()
        clusters = song_db['cluster'].tolist()
        
        print("✅ Preprocessed data loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Model file not found: {e}")
        print("Please run 'python run_preprocessing.py' locally and commit the 'models/' and 'music_dataset.csv' files.")
        raise e
    except Exception as e:
        print(f"❌ Error loading preprocessed data: {e}")
        raise e

def get_collaborative_recommendations(user_id, num_recommendations=5):
    """Get collaborative filtering recommendations"""
    predictions = []
    for song_id in range(len(valid_files)):
        pred = svd.predict(user_id, song_id)
        predictions.append((song_id, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in predictions[:num_recommendations]]

def find_song_index(song_name: str) -> int:
    """Find song index by name (case-insensitive partial match)"""
    song_name_lower = song_name.lower()
    for i, file_name in enumerate(valid_files):
        if song_name_lower in file_name.lower():
            return i
    raise HTTPException(status_code=404, detail=f"Song '{song_name}' not found")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        total_songs=len(valid_files) if valid_files else 0,
        loaded=similarity_matrix is not None
    )

@app.get("/songs")
async def get_all_songs():
    """Get list of all available songs"""
    return {
        "songs": [
            {"index": i, "name": song, "mood": moods[i], "cluster": int(clusters[i])} 
            for i, song in enumerate(valid_files)
        ]
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_songs(request: RecommendationRequest):
    """
    Get hybrid music recommendations based on a seed song and user preferences
    """
    # Find the query index
    if request.song_index is not None:
        if request.song_index >= len(valid_files):
            raise HTTPException(status_code=400, detail="Invalid song index")
        query_index = request.song_index
    elif request.song_name is not None:
        query_index = find_song_index(request.song_name)
    else:
        raise HTTPException(status_code=400, detail="Either song_index or song_name must be provided")

    # Validate user_id
    if request.user_id < 1 or request.user_id > 5:
        raise HTTPException(status_code=400, detail="User ID must be between 1 and 5")

    # Get content-based scores
    content_scores = list(enumerate(similarity_matrix[query_index]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True if request.same_cluster else False)

    # Get collaborative recommendations
    collab_recs = get_collaborative_recommendations(request.user_id, request.num_recommendations * 2)

    # Hybrid blending
    hybrid_weight = 0.5  # You can make this configurable
    query_cluster = clusters[query_index]
    query_mood = moods[query_index]

    hybrid_candidates = {}
    for idx, content_score in content_scores:
        if idx == query_index:
            continue
        if request.same_cluster and clusters[idx] != query_cluster:
            continue
        if request.mood_filter and moods[idx] != request.mood_filter:
            continue
            
        hybrid_score = (hybrid_weight * content_score + 
                        (1 - hybrid_weight) * (5 if idx in collab_recs else 0))
        hybrid_candidates[idx] = hybrid_score

    # Get top recommendations
    top_indices = sorted(hybrid_candidates, key=hybrid_candidates.get, reverse=True)[:request.num_recommendations]

    recommendations = []
    for idx in top_indices:
        recommendations.append(SongRecommendation(
            song_name=valid_files[idx],
            mood=moods[idx],
            cluster=int(clusters[idx]),
            score=float(hybrid_candidates[idx])
        ))

    return RecommendationResponse(
        seed_song=valid_files[query_index],
        recommendations=recommendations
    )

@app.get("/recommend/random")
async def recommend_random_songs(user_id: int = 1, num_recommendations: int = 5):
    """Get recommendations based on a random seed song"""
    import random
    random_index = random.randint(0, len(valid_files) - 1)
    
    request = RecommendationRequest(
        song_index=random_index,
        user_id=user_id,
        num_recommendations=num_recommendations
    )
    
    return await recommend_songs(request)

@app.get("/song/{song_index}")
async def get_song_details(song_index: int):
    """Get detailed information about a specific song"""
    if song_index >= len(valid_files):
        raise HTTPException(status_code=404, detail="Song index not found")
    
    return {
        "index": song_index,
        "name": valid_files[song_index],
        "mood": moods[song_index],
        "cluster": int(clusters[song_index]),
        "features": {
            "tempo": float(song_db.iloc[song_index]['tempo']),
            "energy": float(song_db.iloc[song_index]['rms']),
            "brightness": float(song_db.iloc[song_index]['centroid'])
        }
    }

