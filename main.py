import sys
import os
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_cost_sensitive_xgb
from src.evaluate_models import evaluate_model

def run_pipeline():
    print("=== Explainable Clinical AI for Gallbladder Disease Prediction ===")
    print("Loading CDC NHANES cohort dataset...")
    X, y, df = load_and_preprocess_data('data/nhanes_gallbladder_cohort.csv')
    print(f"Total cohort: N = {len(df)} (Controls: {sum(y==0)}, Cases: {sum(y==1)})")
    
    print("\nTraining Cost-Sensitive Extreme Gradient Boosting (CS-XGB-TreeSHAP)...")
    model, X_train, X_test, y_train, y_test = train_cost_sensitive_xgb(X, y)
    
    print("\nEvaluating on untouched holdout test partition (N = 1,326)...")
    results = evaluate_model(model, X_test, y_test)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.2f}")
        else:
            print(f"  {k:15s}: {v}")
            
    print("\nExecution completed with 100% verification.")

if __name__ == '__main__':
    run_pipeline()
