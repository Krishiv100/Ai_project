# QCAS-DF Full-Stack Website

A deployable frontend + FastAPI inference backend for the QCAS-DF deepfake detector.

## Project structure

```text
QCAS_DF_FULLSTACK_WEBSITE/
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── config.js
│   └── app.js
└── backend/
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    ├── .dockerignore
    ├── .env.example
    ├── qcas_df/
    │   ├── __init__.py
    │   ├── model.py
    │   └── utils.py
    └── model/
        ├── best.pt              ← YOU ADD THIS
        └── calibration.json     ← YOU ADD THIS
```

## 1. Add your trained model

From Kaggle, download your final results package or these two files:

```text
/kaggle/working/QCAS_DF/runs/qcas_df/best.pt
/kaggle/working/QCAS_DF/runs/qcas_df/calibration.json
```

Put them in:

```text
backend/model/
```

## 2. Run locally

### Backend

Open a terminal:

```bash
cd backend
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Start API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Check:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

### Frontend

Open another terminal:

```bash
cd frontend
python -m http.server 5500
```

Open:

```text
http://localhost:5500
```

The default `frontend/config.js` already points to:

```text
http://localhost:8000
```

## 3. Deploy frontend

The frontend is completely static, so you can deploy the `frontend/` folder on:

- GitHub Pages
- Cloudflare Pages
- Netlify
- Vercel static hosting
- any normal web server

After deploying the backend, edit:

```js
// frontend/config.js
window.QCAS_API_BASE = "https://YOUR-BACKEND-DOMAIN";
```

Then redeploy the frontend.

## 4. Deploy backend

The backend is a normal FastAPI application and includes a Dockerfile.

You can deploy it on any host that can run Python/Docker, such as:

- Render
- Railway
- Fly.io
- Google Cloud Run
- AWS
- Azure
- your own VPS/server

For production, set the environment variable:

```text
FRONTEND_ORIGINS=https://YOUR-FRONTEND-DOMAIN
```

The backend starts with:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 5. Docker

From `backend/`:

```bash
docker build -t qcas-df-api .
docker run --rm -p 8000:8000 \
  -e FRONTEND_ORIGINS=http://localhost:5500 \
  qcas-df-api
```

## API

### Health

```http
GET /health
```

### Analyze image

```http
POST /api/analyze/image
Content-Type: multipart/form-data
file=<image>
```

### Analyze video

```http
POST /api/analyze/video
Content-Type: multipart/form-data
file=<video>
```

## Notes

- The backend uses MTCNN to detect the main face before QCAS-DF inference.
- Video inference uniformly samples frames and aggregates face-level fake probabilities.
- CPU deployment works, but video inference may be slow.
- A GPU backend is optional and will improve latency.
- This is a research prototype; predictions should not be treated as definitive forensic proof.
