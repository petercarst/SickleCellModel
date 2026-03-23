from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import requests

app = FastAPI()

# Loading model
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
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {"instances": img_batch.tolist()}
    response = requests.post(endpoint, json=json_data)
    response.raise_for_status()

    # Get single sigmoid output
    prediction = float(response.json()["predictions"][0][0])  # single probability
    predicted_class = "Sickle" if prediction >= 0.5 else "Normal"

    return {
        "class": predicted_class,
        "confidence": prediction if prediction >= 0.5 else 1 - prediction
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)