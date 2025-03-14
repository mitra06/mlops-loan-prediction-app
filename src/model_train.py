import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class LoanModelTrainer:
    def __init__(self, data_path: str, model_path: str, experiment_name="Loan_Prediction_Experiment"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Set up MLflow
        mlflow.set_tracking_uri("file:///D:/MLOPS/mlops-loan-prediction-project/mlruns")
        mlflow.set_experiment(experiment_name)

    def load_data(self) -> pd.DataFrame:
        """Load processed data."""
        return pd.read_csv(self.data_path)

    def split_data(self, df: pd.DataFrame):
        """Split data into train and test sets."""
        df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")
        X = df.drop(columns=["Loan_Status"])
        y = df["Loan_Status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

        return accuracy, report

    def save_model(self):
        """Save the trained model."""
        joblib.dump(self.model, self.model_path)
        print(f"Model saved at: {self.model_path}")

    def train_with_mlflow(self):
        """Train and track model using MLflow."""
        with mlflow.start_run():
            df = self.load_data()
            X_train, X_test, y_train, y_test = self.split_data(df)

            # Log parameters
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("random_state", 42)

            # Train model
            self.train_model(X_train, y_train)

            # Evaluate model
            accuracy, report = self.evaluate_model(X_test, y_test)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", report["weighted avg"]["precision"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])
            mlflow.log_metric("f1-score", report["weighted avg"]["f1-score"])

            # Save & log model
            self.save_model()
            mlflow.sklearn.log_model(self.model, "loan_model")

if __name__ == "__main__":
    # Preprocess data first (if not already processed)
    #preprocessor = DataPreprocessor("data/raw/loan_data.csv")
    #df_processed = preprocessor.preprocess_data()
    #preprocessor.save_data(df_processed, "data/processed/loan_data.csv")

    # Train and track model with MLflow
    trainer = LoanModelTrainer("data/processed/loan_data.csv", "models/loan_model.pkl")
    trainer.train_with_mlflow()
