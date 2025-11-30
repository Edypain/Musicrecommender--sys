from fastapi.testclient import TestClient
from api.recommend import app


def test_health():
    # Create client inside the test so FastAPI startup events run and models are loaded
    with TestClient(app) as client:
        resp = client.get('/')
        assert resp.status_code == 200
        data = resp.json()
        assert 'status' in data
        assert 'total_songs' in data


def test_recommend_random():
    with TestClient(app) as client:
        # request a small number of recommendations
        resp = client.get('/recommend/random', params={'user_id': 1, 'num_recommendations': 2})
        assert resp.status_code == 200
        data = resp.json()
        # The dummy implementation returns seed_song and recommendations list
        assert 'seed_song' in data
        assert 'recommendations' in data
        assert isinstance(data['recommendations'], list)
        # each recommendation should have the expected fields
        if len(data['recommendations']) > 0:
            rec = data['recommendations'][0]
            assert 'song_name' in rec
            assert 'score' in rec
