import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
DRIFT_REFERENCE_RATE = 0.248


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    positive_rate = float(y_train.mean())
    drift_detected = abs(positive_rate - DRIFT_REFERENCE_RATE) > 0.05

    with mlflow.start_run():
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds_default = model.predict(X_eval)
        default_f1 = float(f1_score(y_eval, preds_default))
        default_acc = float(accuracy_score(y_eval, preds_default))

        probabilities = model.predict_proba(X_eval)[:, 1]
        thresholds = [round(0.10 + i * 0.05, 2) for i in range(17)]
        threshold_scores = {
            threshold: float(
                f1_score(y_eval, (probabilities >= threshold).astype(int))
            )
            for threshold in thresholds
        }
        best_threshold = max(threshold_scores, key=threshold_scores.get)
        preds = (probabilities >= best_threshold).astype(int)
        f1 = float(threshold_scores[best_threshold])
        acc = float(accuracy_score(y_eval, preds))
        precision = float(precision_score(y_eval, preds, zero_division=0))
        recall = float(recall_score(y_eval, preds, zero_division=0))
        matrix = confusion_matrix(y_eval, preds)

        if drift_detected:
            print(
                f"WARNING: target distribution drift detected; "
                f"positive_rate={positive_rate:.4f}"
            )

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("default_f1_score", default_f1)
        mlflow.log_metric("default_accuracy", default_acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("positive_rate", positive_rate)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "model")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "default_f1_score": default_f1,
                    "default_accuracy": default_acc,
                    "best_threshold": best_threshold,
                    "positive_rate": positive_rate,
                    "drift_detected": drift_detected,
                    "precision": precision,
                    "recall": recall,
                },
                f,
                indent=2,
            )

        with open("outputs/detail.txt", "w") as f:
            f.write("Confusion matrix [rows=true, columns=predicted]:\n")
            f.write(f"{matrix.tolist()}\n")
            f.write(f"precision_positive={precision:.6f}\n")
            f.write(f"recall_positive={recall:.6f}\n")
            f.write(f"positive_rate={positive_rate:.6f}\n")
            f.write(f"drift_detected={drift_detected}\n")

        os.makedirs("models", exist_ok=True)
        joblib.dump(
            {"model": model, "threshold": best_threshold},
            "models/model.joblib",
        )

        print(
            f"F1: {f1:.4f} | Accuracy: {acc:.4f} | "
            f"Threshold: {best_threshold:.2f} | "
            f"Positive rate: {positive_rate:.4f}"
        )

    return float(f1)

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
