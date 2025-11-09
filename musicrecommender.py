from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import uvicorn
import tempfile
import urllib.request
import json

app = FastAPI(
    title="Music Recommender API",
    description="A hybrid music recommendation system using content-based and collaborative filtering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
feature_matrix = None
similarity_matrix = None
song_db = None
scaler = None
pca = None
kmeans = None
svd = None
valid_files = None
moods = None
clusters = None

# Pydantic models (keep your existing models)
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

# Sample data for deployment (since you can't process audio on Vercel)
def create_sample_data():
    """Create sample data for demonstration on Vercel"""
    global feature_matrix, similarity_matrix, song_db, valid_files, moods, clusters
    
    # Create sample songs
    sample_songs = [
        "song_1.mp3", "song_2.mp3", "song_3.mp3", "song_4.mp3", "song_5.mp3",
        "song_6.mp3", "song_7.mp3", "song_8.mp3", "song_9.mp3", "song_10.mp3"
    ]
    
    # Create sample features
    np.random.seed(42)
    feature_matrix = np.random.rand(len(sample_songs), 20)
    
    # Create similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Sample moods and clusters
    mood_options = ["Energetic", "Calm", "Balanced"]
    moods = [np.random.choice(mood_options) for _ in sample_songs]
    clusters = np.random.randint(0, 3, len(sample_songs))
    
    valid_files = sample_songs
    
    # Create sample song database
    song_db = pd.DataFrame({
        'file': sample_songs,
        'tempo': np.random.uniform(60, 180, len(sample_songs)),
        'zcr': np.random.uniform(0, 0.5, len(sample_songs)),
        'centroid': np.random.uniform(1000, 5000, len(sample_songs)),
        'rolloff': np.random.uniform(2000, 8000, len(sample_songs)),
        'rms': np.random.uniform(0.01, 0.2, len(sample_songs)),
        'bandwidth': np.random.uniform(1000, 4000, len(sample_songs)),
        'onset': np.random.uniform(0.1, 0.9, len(sample_songs)),
        'cluster': clusters,
        'mood': moods
    })
    
    print("✅ Sample data created for demonstration")

@app.on_event("startup")
async def startup_event():
    """Initialize with sample data for Vercel deployment"""
    global feature_matrix, similarity_matrix, song_db, valid_files, moods, clusters
    
    try:
        # On Vercel, we use sample data since we can't process audio files
        create_sample_data()
        print(f"✅ Music Recommender initialized with {len(valid_files)} sample songs")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        # Don't raise error, continue with sample data
        create_sample_data()

def get_collaborative_recommendations(user_id, num_recommendations=5):
    """Get sample collaborative recommendations"""
    # For demo purposes, return random recommendations
    np.random.seed(user_id)
    return np.random.choice(len(valid_files), num_recommendations, replace=False).tolist()

def find_song_index(song_name: str) -> int:
    """Find song index by name"""
    song_name_lower = song_name.lower()
    for i, file_name in enumerate(valid_files):
        if song_name_lower in file_name.lower():
            return i
    # Return a default index if not found
    return 0

# Keep your existing endpoints but remove audio processing
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        total_songs=len(valid_files) if valid_files else 0,
        loaded=feature_matrix is not None
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

    # Get content-based scores
    content_scores = list(enumerate(similarity_matrix[query_index]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)

    # Get collaborative recommendations
    collab_recs = get_collaborative_recommendations(request.user_id, request.num_recommendations * 2)

    # Hybrid blending
    hybrid_weight = 0.7
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
            
        collab_score = 1 if idx in collab_recs else 0
        hybrid_score = (hybrid_weight * content_score + (1 - hybrid_weight) * collab_score)
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

# For Vercel deployment
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)