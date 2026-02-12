import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report




def load_sample_data():
    """Try to load a sample diabetes dataset shipped with the repo or from sklearn."""
    try:
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame
        # Note: sklearn's diabetes dataset is regression; user should provide classification data.
        return df
    except Exception:
        return None

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
    'accuracy': float(accuracy_score(y_test, preds)),
    'f1_score': float(f1_score(y_test, preds, average='weighted')),
    }
    # roc_auc only for binary
    try:
        if len(set(y_test)) == 2:
            probs = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_test, probs))
    except Exception:
        pass

    return metrics