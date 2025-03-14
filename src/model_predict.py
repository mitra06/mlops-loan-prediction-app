from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Dict
import numpy as np
from sklearn.impute import SimpleImputer  # ✅ ADD THIS IMPORT

app = FastAPI()

# Load trained model and preprocessing components
model = joblib.load("models/loan_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
scaler = joblib.load("models/scaler.pkl")

def preprocess_input(input_data: Dict) -> pd.DataFrame:
    """Preprocess API input before prediction."""
    input_df = pd.DataFrame([input_data])

    # Define categorical and numerical columns
    categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
    numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

    # ✅ Apply SimpleImputer to handle missing values
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    input_df[numerical_cols] = num_imputer.fit_transform(input_df[numerical_cols])
    input_df[categorical_cols] = cat_imputer.fit_transform(input_df[categorical_cols])

    # ✅ Apply label encoding using saved encoders
    for col in categorical_cols:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
        else:
            raise ValueError(f"Unexpected value in {col}")

    # ✅ Scale numerical features using saved scaler
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    return input_df

class LoanRequest(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int
    Property_Area: str

@app.post("/predict/")
async def predict_loan(request: LoanRequest):
    try:
        input_data = request.dict()
        processed_df = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(processed_df)

        return {"Loan_Status": "Approved" if prediction[0] == 1 else "Rejected"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
