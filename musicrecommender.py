import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

get_ipython().system('unzip /content/Music_folder.zip')




Audio_dir = "/content/Music_folder" #directory path
Sample_rate = 22050
Duration = 30 # analyze first 30 songs


def extract_features(file_path):
  #Extract 6 key audio features from audio file
  try:
    y, sr = librosa.load(file_path, sr=Sample_rate, duration=Duration)

    #texture or capt timbre
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # chroma(harmonic content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    #spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    #tempo (beats/min)
    tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)[0]

    #Zero crossing rate(noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    #spectral centroid(brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)

    #spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)

    #combining all features into a vector
    return np.hstack((mfcc_mean, chroma_mean, contrast_mean, [tempo, zcr_mean, centroid_mean], rolloff_mean))
  except Exception as e:
    print(f"Error extracting features from {file_path}: {e}")
    return None

#the main pipeline, step1:Extract features for all audio files
audio_files =[os.path.join(Audio_dir, f) for f in os.listdir(Audio_dir) if f.endswith(('.mp3','.wav'))] #  file extension filter

features = []
valid_files = []

for file in audio_files:
  feat = extract_features(file)
  if feat is not None:
    features.append(feat)
    valid_files.append(os.path.basename(file))

    #step 2 create feature matrix and normalize
if features: # Added check to ensure features is not empty
    feature_matrix = np.array(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    #step 3 calculate cosine similarity
    similarity_matrix = cosine_similarity(scaled_features)

    #step 4 recommendation function
    def recommend_songs(query_index, num_recommendations=5):
      sim_scores = list(enumerate(similarity_matrix[query_index]))
      sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)

      #Get top recommendations skipping the query itself
      top_indices=[i[0] for i in sim_scores[1:num_recommendations+1]]

      #display results
      print(f"\nRecommended songs for '{valid_files[query_index]}':")
      for i, idx in enumerate(top_indices):
        print(f"{i+1}.{valid_files[idx]} (similarity:{sim_scores[idx][1]:.3f})")


    #step 5 create song database for reference
    song_db = pd.DataFrame({
        'file':valid_files,
        'tempo':feature_matrix[:, -4], #tempo is 4th from end based on the order in hstack
        'zcr':feature_matrix[:, -3], #zcr is 3rd from end
        'centroid':feature_matrix[:, -2], #centroid is 2th from end
        'rolloff':feature_matrix[:, -1] # rolloff is last
    })


    def build_dataset(tracks):
      dataset=[]

      for track in tracks:
        #Extract features from the audio file
        features = extract_features(track['path'])

        if features is not None:
          #create a row with id and features
          row = {
              'id':track['id'],
              'features':features,
              'title':track['title'],
              'artist':'Unknown Artist', # Added placeholder
              'album':'Unknown Album',  # Added placeholder
              'duration':0,           # Added placeholder
              'genre':'Unknown Genre' # Added placeholder

          }

          feature_names =[
              'mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13',
              'chroma1','chroma2','chroma3','chroma4','chroma5','chroma6','chroma7','chroma8','chroma9','chroma10','chroma11','chroma12',
              'contrast1','contrast2','contrast3','contrast4','contrast5','contrast6','contrast7',
              'tempo','zcr','centroid','rolloff'
          ]
          for i, value in enumerate(features):
            if i < len(feature_names):
              row[feature_names[i]]= value

            else:
              row[f'feature_{i}']= value

          dataset.append(row)
        else:
            print(f"Skipping track {track['id']} due to processing error")
            # Removed return dataset from here

      return dataset 


    if __name__=="__main__":

      music_directory = "/content/Music_folder" # corrected path

      tracks =[]
      for i, filename in enumerate(os.listdir(music_directory)):
        if filename.endswith(('.mp3', '.wav','.ogg','.filac')):
          tracks.append({
              'id':i,
              'path':os.path.join(music_directory, filename),
              'title':os.path.splitext(filename)[0],})

      
      dataset = build_dataset(tracks)

      if dataset: # Added check if dataset is not empty
          df = pd.DataFrame(dataset)
          df.to_csv('music_dataset.csv', index=False)
          print(f"Dataset created with {len(df)} tracks")
      else:
          print("No valid audio files found or features could not be extracted.")


      print("Successfully processed", len(valid_files), "songs")
      print("feature vector length:", feature_matrix.shape[1])
      print("\nSong database:")
      print(song_db.head())

      recommend_songs(query_index=0) 
else:
    print("No valid audio files found or features could not be extracted.")



recommend_songs(query_index=3, num_recommendations=2)
