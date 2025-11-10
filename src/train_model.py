# src/train_model.py
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
import joblib
import json
from datetime import datetime
import os

def train_final_model():
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()
    
    model = Lasso(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    
    print(f"MAE en test: {mae:,.1f} kW")
    
    # Guardar
    version = "v1.0.0"
    path = f'../models/modelo_{version}.pkl'
    joblib.dump(model, path)
    
    # Registry
    registry = {
        "version": version,
        "mae": mae,
        "path": path,
        "date": datetime.now().isoformat()
    }
    with open('../models/model_registry.json', 'w') as f:
        json.dump(registry, f, indent=2)
    
    print("Modelo final guardado")

if __name__ == "__main__":
    train_final_model()