import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import seaborn as sns
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

Audio_dir = "/content/Music_folder"  # directory path
Sample_rate = 22050
Duration = 30  # analyze first 30 seconds of each song

def extract_features(file_path):
    # Extract enhanced set of audio features from audio file
    try:
        y, sr = librosa.load(file_path, sr=Sample_rate, duration=Duration)

        # Texture or timbre
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Chroma (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # Tempo (beats/min)
        tempo = librosa.feature.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)[0]

        # Zero crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)

        # New: RMS (energy level)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # New: Spectral bandwidth (frequency spread)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)

        # New: Onset strength (rhythm complexity)
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = np.mean(onset)

        # Combining all features into a vector
        return np.hstack((mfcc_mean, chroma_mean, contrast_mean, [tempo, zcr_mean, centroid_mean, rolloff_mean, rms_mean, bandwidth_mean, onset_mean]))
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def process_file(file_path):
    feat = extract_features(file_path)
    if feat is not None:
        return feat, os.path.basename(file_path)
    return None, None

# Main pipeline
# Step 1: Extract features for all audio files in parallel
audio_files = [os.path.join(Audio_dir, f) for f in os.listdir(Audio_dir) if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))]  # Fixed typo in '.filac'

with Pool(processes=cpu_count()) as pool:
    results = pool.map(process_file, audio_files)

features = [feat for feat, _ in results if feat is not None]
valid_files = [file for _, file in results if file is not None]

if features:
    # Step 2: Create feature matrix and normalize
    feature_matrix = np.array(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # New: Dimensionality reduction with PCA
    pca = PCA(n_components=0.95)  # Retain 95% variance
    reduced_features = pca.fit_transform(scaled_features)
    print(f"Reduced feature dimensions: {reduced_features.shape[1]}")

    # Step 3: Calculate cosine similarity on reduced features
    similarity_matrix = cosine_similarity(reduced_features)

    # New: Clustering with k-means for mood groups
    num_clusters = min(5, len(valid_files))  # Up to 5 clusters
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    clusters = kmeans.fit_predict(reduced_features)

    # New: Mood estimation heuristic
    def estimate_mood(tempo, rms_mean):
        if tempo > 120 and rms_mean > 0.1:
            return "Energetic"
        elif tempo < 80 and rms_mean < 0.05:
            return "Calm"
        else:
            return "Balanced"

    moods = [estimate_mood(feature_matrix[i, -7], feature_matrix[i, -3]) for i in range(len(valid_files))]  # tempo is -7, rms -3 in hstack

    # New: Collaborative Filtering Setup (using Surprise library)
    # Simulate user ratings (in a real app, load from database)
    # For demo, create a sample user-item ratings DataFrame
    user_ratings = pd.DataFrame({
        'user_id': np.repeat(range(1, 6), len(valid_files)),  # 5 users
        'song_id': np.tile(range(len(valid_files)), 5),
        'rating': np.random.randint(1, 6, size=5 * len(valid_files))  # Random ratings 1-5
    })

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_ratings[['user_id', 'song_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    svd = SVD()
    svd.fit(trainset)

    def get_collaborative_recommendations(user_id, num_recommendations=5):
        predictions = []
        for song_id in range(len(valid_files)):
            pred = svd.predict(user_id, song_id)
            predictions.append((song_id, pred.est))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in predictions[:num_recommendations]]

    # Step 4: Hybrid recommendation function (content + collaborative + clustering/mood)
    def recommend_songs(query_index, user_id, num_recommendations=5, same_cluster=True, mood_filter=None, hybrid_weight=0.5):
        query_cluster = clusters[query_index]
        query_mood = moods[query_index]

        # Content-based scores
        content_scores = list(enumerate(similarity_matrix[query_index]))
        content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True if same_cluster else False)

        # Collaborative scores
        collab_recs = get_collaborative_recommendations(user_id, num_recommendations * 2)  # Get more for blending

        # Hybrid: Blend content and collaborative
        hybrid_candidates = {}
        for idx, content_score in content_scores:
            if idx == query_index:
                continue
            if same_cluster and clusters[idx] != query_cluster:
                continue
            if mood_filter and moods[idx] != mood_filter:
                continue
            hybrid_score = hybrid_weight * content_score + (1 - hybrid_weight) * (5 if idx in collab_recs else 0)  # Normalize collab to 0-5 scale
            hybrid_candidates[idx] = hybrid_score

        top_indices = sorted(hybrid_candidates, key=hybrid_candidates.get, reverse=True)[:num_recommendations]

        print(f"\nHybrid Recommendations for '{valid_files[query_index]}' (User {user_id}, Mood: {query_mood}, Cluster: {query_cluster}):")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {valid_files[idx]} (Mood: {moods[idx]}, Hybrid Score: {hybrid_candidates[idx]:.3f})")

    # New: Generate progressive playlist with hybrid
    def generate_progressive_playlist(query_index, user_id, length=10):
        half = length // 2
        recommend_songs(query_index, user_id, half, same_cluster=True)
        print("\nTransition to contrasting songs:")
        recommend_songs(query_index, user_id, half, same_cluster=False)

    # Step 5: Create enhanced song database
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

    # New: Visualization
    def visualize_similarity():
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False, cmap='YlGnBu')
        plt.title('Song Similarity Heatmap')
        plt.savefig('similarity_heatmap.png')
        print("Similarity heatmap saved as 'similarity_heatmap.png'")

    def visualize_features():
        plt.figure(figsize=(12, 6))
        song_db[['tempo', 'rms', 'bandwidth']].boxplot()
        plt.title('Feature Distribution')
        plt.savefig('feature_distribution.png')
        print("Feature distribution saved as 'feature_distribution.png'")

    # Build dataset function (enhanced with mood and cluster)
    def build_dataset(tracks):
        dataset = []
        for track in tracks:
            features = extract_features(track['path'])
            if features is not None:
                row = {
                    'id': track['id'],
                    'features': features.tolist(),  # Store as list for CSV
                    'title': track['title'],
                    'artist': 'Unknown Artist',
                    'album': 'Unknown Album',
                    'duration': 0,
                    'genre': 'Unknown Genre'
                }
                feature_names = [
                    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13',
                    'chroma1', 'chroma2', 'chroma3', 'chroma4', 'chroma5', 'chroma6', 'chroma7', 'chroma8', 'chroma9', 'chroma10', 'chroma11', 'chroma12',
                    'contrast1', 'contrast2', 'contrast3', 'contrast4', 'contrast5', 'contrast6', 'contrast7',
                    'tempo', 'zcr', 'centroid', 'rolloff', 'rms', 'bandwidth', 'onset'
                ]
                for i, value in enumerate(features):
                    if i < len(feature_names):
                        row[feature_names[i]] = value
                    else:
                        row[f'feature_{i}'] = value
                dataset.append(row)
        return dataset

if __name__ == "__main__":
    music_directory = "/content/Music_folder"

    tracks = []
    for i, filename in enumerate(os.listdir(music_directory)):
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            tracks.append({
                'id': i,
                'path': os.path.join(music_directory, filename),
                'title': os.path.splitext(filename)[0],
            })

    dataset = build_dataset(tracks)

    if dataset:
        df = pd.DataFrame(dataset)
        df.to_csv('music_dataset.csv', index=False)
        print(f"Dataset created with {len(df)} tracks")
    else:
        print("No valid audio files found or features could not be extracted.")

    if features:
        print("Successfully processed", len(valid_files), "songs")
        print("Feature vector length:", feature_matrix.shape[1])
        print("\nSong database:")
        print(song_db.head())

        # Example usage: Hybrid recommendation for user 1 on song 0
        recommend_songs(query_index=0, user_id=1, num_recommendations=5)

        # Generate progressive playlist for user 3 on song 3
        generate_progressive_playlist(query_index=3, user_id=3, length=6)

        # Visualizations
        visualize_similarity()
        visualize_features()
else:
    print("No valid audio files found or features could not be extracted.")