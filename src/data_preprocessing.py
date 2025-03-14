import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os

class DataPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Load dataset from a CSV file."""
        return pd.read_csv(self.file_path)

    def preprocess_numerical(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        """Handle missing values & scale numerical features."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
            ("scaler", self.scaler)  # Normalize numerical features
        ])
        df[numerical_cols] = num_pipeline.fit_transform(df[numerical_cols])
        return df

    def preprocess_categorical(self, df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """Apply Label Encoding to categorical features."""
        for col in categorical_cols:
            if col == "Dependents":  # Special case for 'Dependents'
                df[col] = df[col].replace({"3+": "3"})
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le  # Store encoder for future use
        return df

    def preprocess_data(self) -> pd.DataFrame:
        """Main function to preprocess data."""
        df = self.load_data()

        # Identify categorical and numerical columns
        categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
        numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

        # Preprocess numerical & categorical features
        df = self.preprocess_numerical(df, numerical_cols)
        df = self.preprocess_categorical(df, categorical_cols)

        return df

    def save_preprocessing_models(self, output_dir: str = "models/"):
        """Save the fitted label encoders and scaler."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        print(f"Preprocessing models saved in {output_dir}")

    def save_data(self, df: pd.DataFrame, output_path: str):
        """Save the processed data to a CSV file."""
        df.to_csv(output_path, index=False, header=True)
        print(f"Preprocessing complete! Processed data saved at: {output_path}")

if __name__ == "__main__":
    # Initialize and run the preprocessing pipeline
    preprocessor = DataPreprocessor("data/raw/loan_data.csv")
    df_processed = preprocessor.preprocess_data()
    
    # Save processed data and preprocessing models
    preprocessor.save_data(df_processed, "data/processed/loan_data.csv")
    preprocessor.save_preprocessing_models()
