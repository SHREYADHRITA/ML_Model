import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, precision_score, \
    recall_score, matthews_corrcoef


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
    """Evaluate a trained classification model using multiple metrics."""
    preds = model.predict(X_test)

    metrics = {
        "Accuracy": round(float(accuracy_score(y_test, preds)),4),
        "Precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4),
        "F1 Score": round(float(f1_score(y_test, preds, average="weighted")), 4),
        "MCC Score": round(float(matthews_corrcoef(y_test, preds)), 4),
    }

    # Compute AUC Score (only for binary classification)
    try:
        if len(set(y_test)) == 2 and hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            metrics["AUC Score"] = round(roc_auc_score(y_test, probs), 4)
    except Exception:
        metrics["AUC Score"] = None

    return metrics
