"""
Huấn luyện mô hình Random Forest cho dự đoán ung thư vú (Benign/Malignant)
Dữ liệu: sklearn.datasets.load_breast_cancer (Wisconsin Breast Cancer)
Kết quả:
- Lưu model: models/model.pkl
- Lưu metadata (feature_names, scaler...): models/meta.json
"""

import os
import json
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

SELECTED_FEATURES = [
    "mean concave points",
    "worst concave points",
    "worst area",
    "mean concavity",
    "worst radius"
]

def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    # Chỉ lấy 5 đặc trưng
    X = df[SELECTED_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: Chuẩn hoá + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

    # Lưu model và metadata
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, "model.pkl"))

    meta = {
        "feature_names": SELECTED_FEATURES,
        "class_map": {
            0: "malignant",
            1: "benign"
        }
    }
    with open("models/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
