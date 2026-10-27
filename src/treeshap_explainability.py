import shap
import matplotlib.pyplot as plt

def generate_shap_explanations(model, X_test, feature_names=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values
