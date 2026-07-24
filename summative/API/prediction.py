from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib

app = FastAPI(title="East Africa Unemployment Prediction API")

# CORS: only allow the Flutter app's origin, only the methods/headers it needs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://your-render-app-name.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Load model, scaler, and expected feature column order once at startup
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

numeric_features = [
    "Year", "GDP growth (annual %)", "GDP per capita (current US$)",
    "Inflation, consumer prices (annual %)",
    "Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)",
    "Population growth (annual %)", "Urban population (% of total population)"
]

class PredictionInput(BaseModel):
    year: int = Field(..., ge=2000, le=2030, description="Year of observation")
    gdp_growth: float = Field(..., ge=-15.0, le=20.0, description="GDP growth (annual %)")
    gdp_per_capita: float = Field(..., ge=100.0, le=5000.0, description="GDP per capita (current US$)")
    inflation: float = Field(..., ge=-10.0, le=60.0, description="Inflation, consumer prices (annual %)")
    labor_force_participation: float = Field(..., ge=50.0, le=95.0, description="Labor force participation rate (%)")
    population_growth: float = Field(..., ge=-5.0, le=6.0, description="Population growth (annual %)")
    urban_population: float = Field(..., ge=5.0, le=90.0, description="Urban population (% of total)")
    country: Literal[
        "Burundi", "Djibouti", "Ethiopia", "Kenya", "Rwanda",
        "Somalia, Fed. Rep.", "South Sudan", "Tanzania", "Uganda"
    ] = Field(..., description="Country")

@app.get("/")
def root():
    return {"message": "East Africa Unemployment Prediction API. Go to /docs for Swagger UI."}

@app.post("/predict")
def predict(data: PredictionInput):
    row = {
        "Year": data.year,
        "GDP growth (annual %)": data.gdp_growth,
        "GDP per capita (current US$)": data.gdp_per_capita,
        "Inflation, consumer prices (annual %)": data.inflation,
        "Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)": data.labor_force_participation,
        "Population growth (annual %)": data.population_growth,
        "Urban population (% of total population)": data.urban_population,
    }
    for col in feature_columns:
        if col.startswith("country_"):
            row[col] = (col == f"country_{data.country}")

    input_df = pd.DataFrame([row])[feature_columns]
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    prediction = model.predict(input_df)[0]
    return {"predicted_unemployment_rate": round(float(prediction), 2)}