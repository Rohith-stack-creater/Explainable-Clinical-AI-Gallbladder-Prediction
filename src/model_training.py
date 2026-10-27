import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_cost_sensitive_xgb(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Compute inverse class weighting
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos  # 8.02x
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
