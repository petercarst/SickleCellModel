# SickleCellModel
Sickle cell image classification using deep learning (DenseNet121 + Keras/TensorFlow)

Key components:
FastAPI – Web server handling API requests
TensorFlow Serving – Serves the trained deep learning model

Project Structure:

SickleCellClassification/
├─ api/
│  └─ tf_serving.py         # FastAPI server
├─ models/
│  └─ sickle-cell/          # SavedModel format
├─ dataset/                 # Training, validation, and test datasets
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ README.md

Prerequisites:
   Python 3.10+
   Docker (for TensorFlow Serving)

Setting Up TensorFlow Serving:
   Assuming your model is located at:
       models/sickle-cell

       Run TensorFlow Serving Docker:
       docker run -p 8501:8501 --name tf_serving_sicklecell \
  -v "C:/Users/Student/Desktop/ml/SickleCellClassification/models/sickle-cell:/models/sickle-cell" \
  -e MODEL_NAME=sickle-cell -t tensorflow/serving

   REST API endpoint:
   http://localhost:8501/v1/models/sickle-cell:predict


Running FastAPI
  Start the API server:
  python -m uvicorn api.tf_serving:app --reload


Testing /predict with Postman
  1. POST request to:
    http://localhost:8000/predict

 2. Body → form-data:
    | Key  | Type | Value      |
    | ---- | ---- | ---------- |
    | file | File | sample.jpg |

 3. Response:
   {
  "class": "Sickle",
  "confidence": 0.92
  }
