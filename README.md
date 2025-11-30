Music Recommender

This repository contains a FastAPI-based music recommender service that combines content-based and collaborative filtering. The API is under `api/recommend.py` and a simple static UI is provided in `index.html`.

Important: This project expects preprocessed model files and a dataset CSV to be present in the repository at:

- `models/svd.pkl`
- `models/similarity_matrix.npy`
- `music_dataset.csv`

Vercel notes
-------------
We provide `vercel.json` and `requirements.txt` so you can deploy to Vercel. Vercel will install the Python dependencies and run the Python serverless functions in `api/`.

Local development
-----------------
1. Create a virtual environment and activate it (PowerShell):

```powershell
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the API locally with uvicorn:

```powershell
uvicorn api.recommend:app --reload --port 8000
```

4. Open `http://localhost:8000` to check health or `http://localhost:8000/docs` for interactive API docs. The simple UI is at `http://localhost:8000/index.html` when served by a static server — for local testing open the `index.html` file directly in your browser and it will call the API at `/api/recommend` when you run uvicorn with the above command.

Deploying to Vercel
-------------------
1. Install the Vercel CLI (optional but helpful):

```powershell
npm i -g vercel
```

2. From the project root, run:

```powershell
vercel login
vercel --prod
```

3. If your project contains the required model files and dataset, the API should be available under the deployment URL at `/api/recommend`.

Notes and caveats
-----------------
- Vercel has cold-starts and serverless limits. If your `models/` folder is large, consider hosting models externally (S3/GCS) and loading them at runtime, or using another hosting provider better suited for heavy ML assets.
- If the startup loader can't find the model files the API will return 503 with an explanation. Add the models and `music_dataset.csv` to the repo before deploying or change the loader to fetch them from remote storage.

Remote model hosting (recommended for large artifacts)
-----------------------------------------------------
If your preprocessed model files are large, don't commit them to the repo. Instead host them on an HTTP(S)-accessible location (for example: a public S3 bucket or a signed URL) and set the `MODEL_BASE_URL` environment variable in Vercel to point at the base URL where the files live.

The service expects these paths under the base URL:

- `models/similarity_matrix.npy`
- `models/svd.pkl` (optional)
- `music_dataset.csv`

Example: if your files are at `https://example-bucket.s3.amazonaws.com/`, set `MODEL_BASE_URL=https://example-bucket.s3.amazonaws.com` in Vercel. On startup the app will try to download missing `similarity_matrix.npy` and `music_dataset.csv` from that URL into the runtime filesystem.

Note: Vercel serverless functions have ephemeral filesystems and execution limits. Downloading small artifacts at cold-start works for modest sizes. For very large models (100s of MBs) use a model-serving service or a VM/container host instead.

Next steps (optional)
---------------------
- Add CI that validates model artifacts exist before deploying.
- Add a lightweight test that calls `/recommend/random` and asserts 200.
