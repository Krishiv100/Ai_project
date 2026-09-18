from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import gc
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

from qcas_df.model import QCASDF


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
CHECKPOINT = MODEL_DIR / "best.pt"
CALIBRATION_FILE = MODEL_DIR / "calibration.json"

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
VIDEO_SAMPLE_FRAMES = int(os.getenv("VIDEO_SAMPLE_FRAMES", "3"))
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "15"))
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "150"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(1)

allowed_origins = [
    x.strip()
    for x in os.getenv("FRONTEND_ORIGINS", "*").split(",")
    if x.strip()
]

app = FastAPI(
    title="QCAS-DF API",
    version="1.0.0",
    description="Inference API for Quality-Conditioned Adaptive Spectral Deepfake Detection.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model = None
detector = None
temperature = 1.0
threshold = 0.5
checkpoint_class_map = None

tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_runtime():
    global model, detector, temperature, threshold, checkpoint_class_map

    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"Missing checkpoint: {CHECKPOINT}. "
            "Copy your trained best.pt into backend/model/."
        )

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = QCASDF().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    checkpoint_class_map = ckpt.get("class_to_idx")

    if CALIBRATION_FILE.exists():
        calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        temperature = float(calibration.get("temperature", 1.0))
        threshold = float(calibration.get("threshold", 0.5))

    detector = MTCNN(keep_all=True, device=DEVICE)


@app.on_event("startup")
def startup():
    try:
        load_runtime()
    except Exception as exc:
        # Keep API alive so /health explains what is missing.
        print("QCAS-DF startup warning:", exc)


def require_model():
    if model is None or detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Put best.pt and calibration.json "
                "inside backend/model/ and restart the backend."
            ),
        )


def image_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    save_kwargs = {"quality": quality} if fmt.upper() in {"JPEG", "WEBP"} else {}
    image.save(buf, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def crop_largest_face(image: Image.Image, margin: float = 0.25):
    require_model()
    image = image.convert("RGB")
    boxes, probs = detector.detect(image)

    if boxes is None or len(boxes) == 0:
        return None, None

    p = np.asarray(probs, dtype=float)
    p = np.nan_to_num(p, nan=-1.0)
    idx = int(np.argmax(p))

    x1, y1, x2, y2 = boxes[idx]
    w = x2 - x1
    h = y2 - y1
    x1 -= margin * w
    x2 += margin * w
    y1 -= margin * h
    y2 += margin * h

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image.width, int(x2))
    y2 = min(image.height, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = image.crop((x1, y1, x2, y2)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    line_w = max(2, min(image.width, image.height) // 220)
    draw.rectangle((x1, y1, x2, y2), outline=(78, 224, 255), width=line_w)

    return crop, preview


def predict_face(face: Image.Image) -> dict:
    require_model()
    x = tf(face).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        out = model(x)
        logits = out["logits"] / temperature
        fake_prob = torch.softmax(logits, dim=1)[0, 1].item()

    prediction = "FAKE" if fake_prob >= threshold else "REAL"
    confidence = fake_prob if prediction == "FAKE" else 1.0 - fake_prob

    return {
        "prediction": prediction,
        "fake_probability": float(fake_prob),
        "confidence": float(confidence),
        "threshold": float(threshold),
        "estimated_degradation": float(out["quality"][0].item()),
        "spatial_gate": float(out["gate"][0, 0].item()),
        "spectral_gate": float(out["gate"][0, 1].item()),
        "frequency_boundary_1": float(out["b1"][0].item()),
        "frequency_boundary_2": float(out["b2"][0].item()),
    }


def uniform_indices(n_frames: int, k: int) -> List[int]:
    if n_frames <= 0:
        return []
    if n_frames <= k:
        return list(range(n_frames))
    return np.linspace(0, n_frames - 1, k).astype(int).tolist()


@app.get("/")
def root():
    return {
        "name": "QCAS-DF API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "checkpoint_exists": CHECKPOINT.exists(),
        "calibration_exists": CALIBRATION_FILE.exists(),
        "threshold": threshold if model is not None else None,
        "class_to_idx": checkpoint_class_map,
    }


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    require_model()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    raw = await file.read()
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than the {MAX_IMAGE_MB} MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode this image.") from exc

    face, preview = crop_largest_face(image)
    if face is None:
        raise HTTPException(status_code=422, detail="No face was detected in the image.")

    result = predict_face(face)
    result["face_crop"] = image_to_data_url(face)
    result["face_detection_preview"] = image_to_data_url(preview)
    result["device"] = str(DEVICE)

    return result


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    require_model()

    suffix = Path(file.filename or "upload.mp4").suffix.lower()

    allowed = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    # Save uploaded video
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        total = 0

        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            total += len(chunk)

            if total > MAX_VIDEO_MB * 1024 * 1024:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)

                raise HTTPException(
                    status_code=413,
                    detail=f"Video larger than {MAX_VIDEO_MB} MB."
                )

            tmp.write(chunk)

        temp_path = Path(tmp.name)

    cap = cv2.VideoCapture(str(temp_path))

    if not cap.isOpened():
        temp_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Could not open video."
        )

    try:

        n_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if n_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Video contains no readable frames."
            )

        # ONLY sample 3 frames instead of decoding entire video
        sample_count = min(
            VIDEO_SAMPLE_FRAMES,
            n_frames
        )

        indices = np.linspace(
            0,
            n_frames - 1,
            sample_count
        ).astype(int)

        frame_results = []
        sample_faces = []

        for frame_index in indices:

            # Jump directly to desired frame
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            ok, frame = cap.read()

            if not ok:
                continue

            # Reduce frame resolution BEFORE MTCNN
            h, w = frame.shape[:2]

            max_side = 480

            if max(h, w) > max_side:

                scale = max_side / max(h, w)

                new_w = max(
                    1,
                    int(w * scale)
                )

                new_h = max(
                    1,
                    int(h * scale)
                )

                frame = cv2.resize(
                    frame,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pil = Image.fromarray(rgb)

            face, _ = crop_largest_face(pil)

            if face is not None:

                result = predict_face(face)

                frame_results.append(result)

                # Keep max 3 previews
                if len(sample_faces) < 3:
                    sample_faces.append(
                        image_to_data_url(
                            face,
                            quality=75
                        )
                    )

            # Release memory after each frame
            del frame
            del rgb
            del pil

            if face is not None:
                del face

            gc.collect()

    finally:

        cap.release()

        temp_path.unlink(
            missing_ok=True
        )

        gc.collect()

    if not frame_results:

        raise HTTPException(
            status_code=422,
            detail=(
                "No usable faces were detected "
                "in the sampled video frames."
            )
        )

    fake_probs = [
        r["fake_probability"]
        for r in frame_results
    ]

    mean_fake = float(
        np.mean(fake_probs)
    )

    prediction = (
        "FAKE"
        if mean_fake >= threshold
        else "REAL"
    )

    confidence = (
        mean_fake
        if prediction == "FAKE"
        else 1.0 - mean_fake
    )

    def mean_value(key):

        return float(
            np.mean([
                r[key]
                for r in frame_results
            ])
        )

    return {

        "prediction":
            prediction,

        "video_fake_probability":
            mean_fake,

        "confidence":
            confidence,

        "threshold":
            float(threshold),

        "faces_analyzed":
            len(frame_results),

        "frames_sampled":
            len(indices),

        "mean_estimated_degradation":
            mean_value(
                "estimated_degradation"
            ),

        "mean_spatial_gate":
            mean_value(
                "spatial_gate"
            ),

        "mean_spectral_gate":
            mean_value(
                "spectral_gate"
            ),

        "mean_b1":
            mean_value(
                "frequency_boundary_1"
            ),

        "mean_b2":
            mean_value(
                "frequency_boundary_2"
            ),

        "sample_faces":
            sample_faces,

        "device":
            str(DEVICE)
    }
