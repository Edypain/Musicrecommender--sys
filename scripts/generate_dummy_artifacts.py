"""
Generate small dummy model artifacts so the API can run for testing.
Creates:
 - models/similarity_matrix.npy
 - models/svd.pkl (a tiny pickled object with a .predict method)
 - music_dataset.csv (4 sample rows)

This is for testing and should be replaced with your real artifacts before production.
"""
import os
import numpy as np
import pickle

os.makedirs('models', exist_ok=True)

# Create a small similarity matrix (4 songs)
sim = np.array([
    [1.0, 0.8, 0.2, 0.1],
    [0.8, 1.0, 0.3, 0.2],
    [0.2, 0.3, 1.0, 0.6],
    [0.1, 0.2, 0.6, 1.0]
], dtype=float)
np_path = os.path.join('models', 'similarity_matrix.npy')
np.save(np_path, sim)
print(f'Wrote {np_path}')

# Create a tiny dummy SVD-like object with a predict(user, item) -> object with .est
class DummyPrediction:
    def __init__(self, est):
        self.est = est

class DummySVD:
    def predict(self, user_id, song_id):
        # deterministic fake score derived from ids
        score = float((int(user_id) * 31 + int(song_id) * 17) % 5)
        return DummyPrediction(score)

svd_path = os.path.join('models', 'svd.pkl')
with open(svd_path, 'wb') as f:
    pickle.dump(DummySVD(), f)
print(f'Wrote {svd_path}')

# Create a minimal dataset CSV
csv_path = 'music_dataset.csv'
with open(csv_path, 'w', encoding='utf-8') as f:
    f.write('file,mood,cluster,tempo,rms,centroid\n')
    f.write('song_a.mp3,happy,0,120.0,0.5,2000.0\n')
    f.write('song_b.mp3,calm,0,95.0,0.3,1500.0\n')
    f.write('song_c.mp3,sad,1,75.0,0.2,1200.0\n')
    f.write('song_d.mp3,energetic,1,140.0,0.8,2600.0\n')
print(f'Wrote {csv_path}')

print('Dummy artifacts generated. Reminder: replace these with your real artifacts before production deploy.')
