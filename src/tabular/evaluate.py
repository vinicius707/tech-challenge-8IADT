import joblib
import pandas as pd

def load_model(path="models/maternal_risk_model.pkl"):
    return joblib.load(path)

def predict(model, input_data: pd.DataFrame):
    return model.predict(input_data)
