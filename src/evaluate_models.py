import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred) * 100,
        'Sensitivity': recall_score(y_test, y_pred) * 100,
        'Specificity': (tn / (tn + fp)) * 100,
        'Precision': precision_score(y_test, y_pred) * 100,
        'F1-Score': f1_score(y_test, y_pred) * 100,
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }
    return metrics
