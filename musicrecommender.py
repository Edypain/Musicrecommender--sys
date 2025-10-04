from fastapi import FastAPI, HTTPException
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
import joblib  # for saving/loading models
import uvicorn


app = FastAPI(
    title="Music Recommender API",
    description="A hybrid music recommendation system using content-based and collaborative filtering",
    version="1.0.0"
)

# Global variables to store your models and data
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

# Pydantic models for request/response
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

@app.on_event("startup")
async def startup_event():
    """Initialize the recommender system when the app starts"""
    global feature_matrix, similarity_matrix, song_db, scaler, pca, kmeans, svd, valid_files, moods, clusters
    
    try:
        # Check if preprocessed data exists
        if os.path.exists('music_dataset.csv') and os.path.exists('models'):
            await load_preprocessed_data()
        else:
            # Run your preprocessing pipeline
            await preprocess_data()
        
        print(f"✅ Music Recommender initialized with {len(valid_files)} songs")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise e

async def preprocess_data():
    """Run your existing preprocessing pipeline"""
    global feature_matrix, similarity_matrix, song_db, scaler, pca, kmeans, svd, valid_files, moods, clusters
    
    Audio_dir = "/content/Music_folder"
    Sample_rate = 22050
    Duration = 30

    def extract_features(file_path):
        try:
            y, sr = librosa.load(file_path, sr=Sample_rate, duration=Duration)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast, axis=1)
            tempo = librosa.feature.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            centroid_mean = np.mean(centroid)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff_mean = np.mean(rolloff)
            rms = librosa.feature.rms(y=y)
            rms_mean = np.mean(rms)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            bandwidth_mean = np.mean(bandwidth)
            onset = librosa.onset.onset_strength(y=y, sr=sr)
            onset_mean = np.mean(onset)

            return np.hstack((mfcc_mean, chroma_mean, contrast_mean, [tempo, zcr_mean, centroid_mean, rolloff_mean, rms_mean, bandwidth_mean, onset_mean]))
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None

    def process_file(file_path):
        feat = extract_features(file_path)
        if feat is not None:
            return feat, os.path.basename(file_path)
        return None, None

    # Process audio files
    audio_files = [os.path.join(Audio_dir, f) for f in os.listdir(Audio_dir) 
                   if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))]
    
    from multiprocessing import Pool, cpu_count
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, audio_files)

    features = [feat for feat, _ in results if feat is not None]
    valid_files = [file for _, file in results if file is not None]

    if not features:
        raise Exception("No valid audio files found")

    # Create feature matrix and normalize
    feature_matrix = np.array(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Dimensionality reduction
    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(scaled_features)

    # Similarity matrix
    similarity_matrix = cosine_similarity(reduced_features)

    # Clustering
    num_clusters = min(5, len(valid_files))
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    clusters = kmeans.fit_predict(reduced_features)

    # Mood estimation
    def estimate_mood(tempo, rms_mean):
        if tempo > 120 and rms_mean > 0.1:
            return "Energetic"
        elif tempo < 80 and rms_mean < 0.05:
            return "Calm"
        else:
            return "Balanced"

    moods = [estimate_mood(feature_matrix[i, -7], feature_matrix[i, -3]) 
             for i in range(len(valid_files))]

    # Collaborative Filtering Setup
    user_ratings = pd.DataFrame({
        'user_id': np.repeat(range(1, 6), len(valid_files)),
        'song_id': np.tile(range(len(valid_files)), 5),
        'rating': np.random.randint(1, 6, size=5 * len(valid_files))
    })

    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_ratings[['user_id', 'song_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    svd = SVD()
    svd.fit(trainset)

    # Create song database
    song_db = pd.DataFrame({
        'file': valid_files,
        'tempo': feature_matrix[:, -7],
        'zcr': feature_matrix[:, -6],
        'centroid': feature_matrix[:, -5],
        'rolloff': feature_matrix[:, -4],
        'rms': feature_matrix[:, -3],
        'bandwidth': feature_matrix[:, -2],
        'onset': feature_matrix[:, -1],
        'cluster': clusters,
        'mood': moods
    })

    # Save models and data for future use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, 'models/pca.pkl')
    joblib.dump(kmeans, 'models/kmeans.pkl')
    joblib.dump(svd, 'models/svd.pkl')
    np.save('models/similarity_matrix.npy', similarity_matrix)
    song_db.to_csv('music_dataset.csv', index=False)
    
    print("✅ Data preprocessing completed and models saved")

async def load_preprocessed_data():
    """Load preprocessed data and models"""
    global feature_matrix, similarity_matrix, song_db, scaler, pca, kmeans, svd, valid_files, moods, clusters
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        pca = joblib.load('models/pca.pkl')
        kmeans = joblib.load('models/kmeans.pkl')
        svd = joblib.load('models/svd.pkl')
        similarity_matrix = np.load('models/similarity_matrix.npy')
        song_db = pd.read_csv('music_dataset.csv')
        
        valid_files = song_db['file'].tolist()
        moods = song_db['mood'].tolist()
        clusters = song_db['cluster'].tolist()
        
        print("✅ Preprocessed data loaded successfully")
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)