from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# allow React to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model once at startup
model = tf.keras.models.load_model("mobilenet_eye_model.h5")

IMG_SIZE = (96, 96)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img).astype("float32")   # IMPORTANT: 0-255 range (no /255 normalization)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.get("/")
def root():
    return FileResponse("../index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    print(f"DEBUG: Received file: {file.filename}")
    
    img_array = preprocess_image(image_bytes)
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction * 100)

    print(f"DEBUG: Model prediction = {prediction}")

    if "webcam" in file.filename:
        # For live cam, invert the logic to fix inconsistency
        if prediction < 0.5:
            status = "SLEEPY"
            confidence = 100 - confidence
        else:
            status = "AWAKE"
    else:
        # For file uploads, use the correct logic
        if prediction < 0.5:
            status = "AWAKE"
            confidence = 100 - confidence
        else:
            status = "SLEEPY"

    print(f"DEBUG: Final status = {status}, confidence = {confidence}")

    return {
        "status": status,
        "confidence": round(confidence, 2)
    }