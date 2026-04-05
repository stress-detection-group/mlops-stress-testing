# Income Prediction API

A machine learning pipeline that predicts whether an individual earns more or less than $50K/year, based on the [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult). Includes model training, experiment tracking, a REST API, and automated model quality tests.

---

## Project Structure

```
├── app.py               # FastAPI prediction endpoint
├── train.py             # Model training + MLflow experiment tracking
├── simulate_data.py     # Generates simulated data for drift testing
├── data/
│   ├── adult.csv        # Raw training data
│   ├── X_test.csv       # Saved test features
│   ├── y_test.csv       # Saved test labels
│   └── simulated_data.csv
├── models/
│   ├── best_model.json  # Name of the selected best model
│   └── *.pkl            # Trained model files
├── tests/
│   ├── recall_test.py
│   ├── bias_test.py
│   ├── robustness_test.py
│   └── drift_test.py
├── Dockerfile
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

Trains five classifiers (Logistic Regression, SVM, KNN, Decision Tree, Random Forest) and tracks each run with MLflow. The best model is selected based on a combined score of recall, fairness, and robustness.

```bash
python train.py
```

To view the MLflow experiment UI:

```bash
mlflow ui
```

---

## Running the API

```bash
uvicorn app:app --reload
```

The API loads the best model automatically from `models/best_model.json`.

### Endpoints

`GET /` — health check, returns active model name

`POST /predict` — returns income prediction (0 = <=50K, 1 = >50K)

Example request body:

```json
{
  "age": 39,
  "workclass": 4,
  "fnlwgt": 77516,
  "education": 9,
  "education_num": 13,
  "marital_status": 2,
  "occupation": 1,
  "relationship": 0,
  "race": 4,
  "sex": 1,
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": 39
}
```

---

## Tests

Run each test individually after training:

```bash
python tests/recall_test.py       # Recall >= 0.5
python tests/bias_test.py         # Demographic parity diff <= 0.2
python tests/robustness_test.py   # Prediction flip rate <= 15%
python tests/drift_test.py        # PSI-based feature drift detection
```

To generate simulated drift data before running the drift test:

```bash
python simulate_data.py
```

---

## Docker

```bash
docker build -t income-prediction .
docker run -p 8000:8000 income-prediction
```

---

## Models

| Model               | Notes                          |
|---------------------|--------------------------------|
| Logistic Regression | Baseline linear classifier     |
| SVM                 | Support vector classifier      |
| KNN                 | K-nearest neighbors            |
| Decision Tree       | Interpretable tree model       |
| Random Forest       | Ensemble, typically best score |

Best model is auto-selected and saved to `models/best_model.json` after each training run.
