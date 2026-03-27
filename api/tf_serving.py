from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import httpx

app = FastAPI()

# TensorFlow Serving endpoint
endpoint = "http://localhost:8501/v1/models/sickle-cell:predict"

CLASS_NAMES = ["Sickle Cell", "Normal"]
IMAGE_SIZE = 224 


def read_file_as_image(data):
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image).astype(np.float32) / 255.0
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        # Prepare request for TF Serving
        json_data = {"instances": img_batch.tolist()}

        # Async request to TF Serving
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(endpoint, json=json_data)
            response.raise_for_status()

        # Get prediction (sigmoid output)
        prediction = float(response.json()["predictions"][0][0])

        # Convert to class
        predicted_class = "Sickle Cell" if prediction >= 0.5 else "Normal"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        return {
            "class": predicted_class,
            "confidence": confidence
        }
       
       # Error handling
    except httpx.RequestError as e:
        return {"error": f"Connection error: {str(e)}"}

    except httpx.HTTPStatusError as e:
        return {"error": f"TF Serving error: {e.response.text}"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)