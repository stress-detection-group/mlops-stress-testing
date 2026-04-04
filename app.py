from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import pandas as pd

app = FastAPI()

# ✅ CORS FIX (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model name
with open('models/best_model.json', 'r') as f:
    best_model_name = json.load(f)['best_model']

# Load model (same as your original — no major change)
with open(f'models/{best_model_name}.pkl', 'rb') as f:
    model = pickle.load(f)


class InputData(BaseModel):
    age: float
    workclass: float
    fnlwgt: float
    education: float
    education_num: float
    marital_status: float
    occupation: float
    relationship: float
    race: float
    sex: float
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: float


@app.get("/")
def home():
    return {
        "message": "Income Prediction API",
        "model": best_model_name
    }


@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.dict()])

        input_df.columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education.num',
            'marital.status', 'occupation', 'relationship', 'race', 'sex',
            'capital.gain', 'capital.loss', 'hours.per.week', 'native.country'
        ]

        prediction = model.predict(input_df)[0]

        return {
            "prediction": int(prediction),
            "model_used": best_model_name
        }

    except Exception as e:
        return {"error": str(e)}