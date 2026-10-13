import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def metrics(y_true, y_pred, y_score):
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
    }


def torch_predict(model, X, device, bs=20000):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            probs.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    p = np.concatenate(probs)
    return (p >= 0.5).astype(int), p


def rf_predict(rf, X):
    p = rf.predict_proba(X)[:, 1]
    return (p >= 0.5).astype(int), p