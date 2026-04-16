# VigilEye — Drowsiness Detection Web App

## Quick Start

### 1. Backend (FastAPI)

```bash
# Install dependencies
pip install fastapi uvicorn tensorflow pillow numpy python-multipart

# Place mobilenet_eye_model.h5 in the same folder as main.py
# Then run:
uvicorn main:app --reload --port 8000
python -m uvicorn main:app --reload --port 8000
python -m uvicorn main:app --port 8000
```

Backend runs at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 2. Frontend

Open http://localhost:8000 in your browser.
No build step needed — pure HTML/CSS/JS.

## File Structure

```
├── index.html              ← Frontend (served by backend)
├── backend/
│   ├── main.py             ← FastAPI backend
│   └── mobilenet_eye_model.h5  ← Your trained model (from Colab)
```

## Demo Login

- Username: `admin`
- Password: `vigil123`

## Flow

1. Login → 2. Accept Terms (10k subset disclaimer) → 3. Upload eye image → Get AWAKE/SLEEPY result

## Model Info

- Architecture: MobileNetV2 (transfer learning)
- Trained on: 10,000 images (subset of 84k MRL Eye Dataset)
- Val Accuracy: 88.56%
- Recall: 90.59%
- Input size: 96×96 RGB
