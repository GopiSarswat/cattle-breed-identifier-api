from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# ------------------------------
# Load the TFLite model
# ------------------------------
try:
    interpreter = tf.lite.Interpreter(model_path="cattle_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import sys
    sys.exit(1)

# ------------------------------
# Define class names (39 breeds)
# ------------------------------
class_names = [
    "Alambadi", "Amritmahal", "Ayrshire", "Banni", "Bargur", "Bhadawari",
    "Brown Swiss", "Deoni", "Gir", "Guernsey", "Hallikan", "Hariana",
    "Holstein Friesian", "Jaffrabadi", "Jersey", "Kangayam", "Kankrej",
    "Kasangod", "Khillari", "Krishna Valley", "Malnad gidda", "Mehsana",
    "Murrah", "Nagori", "Nagpuri", "Nimari", "Ongole", "Pulikulam",
    "Rathi", "Red Dane", "Red Sindhi", "Sahival", "Surti", "Tharparkan",
    "Toda", "Umblachery", "Vechur"
]

# ------------------------------
# FastAPI app setup
# ------------------------------
app = FastAPI(
    title="Cattle Breed Classifier API",
    description="Upload an image to classify cattle/buffalo breeds using TensorFlow Lite.",
    version="1.0"
)

# Enable CORS (for frontend/mobile app integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Helper function for preprocessing
# ------------------------------
def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Match training size
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ------------------------------
# Root endpoint
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Cattle Breed Classifier API</title>
        </head>
        <body>
            <h1>🐄 Cattle Breed Classifier API</h1>
            <p>This API classifies cattle/buffalo breeds from images using TensorFlow Lite.</p>
            <ul>
                <li><strong>API Documentation:</strong> <a href="/docs">Swagger UI</a></li>
                <li><strong>Alternative Docs:</strong> <a href="/redoc">ReDoc</a></li>
                <li><strong>Prediction Endpoint:</strong> POST /predict</li>
            </ul>
        </body>
    </html>
    """

# ------------------------------+
# Prediction endpoint
# ------------------------------
@app.post(
    "/predict",
    summary="Predict cattle breed from an image",
    description="Upload a JPG/PNG image and get top-2 breed predictions.",
)
async def predict(file: UploadFile = File(..., description="Upload cattle/buffalo image")):
    # Validate input file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get top 2 predictions
        top_indices = predictions.argsort()[-2:][::-1]
        results = []
        for i in top_indices:
            confidence_percent = float(predictions[i]) * 100
            results.append({
                "breed": class_names[i],
                "confidence": round(confidence_percent, 2)
            })

        return JSONResponse(content={
            "success": True,
            "predictions": results,
            "top_prediction": results[0] if results else None
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ------------------------------
# Health check endpoint
# ------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}
