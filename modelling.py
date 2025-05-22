import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

# menyimpan eksperimen ke Tracking UI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Agar data hasil pelatihan dapat tersimpan pada satu pipeline
mlflow.set_experiment("Latihan Credit Scoring")

# Load Dataset
data = pd.read_csv("https://raw.githubusercontent.com/dicodingacademy/ML-System/refs/heads/main/Modul%202%20-%20Membangun%20dan%20Mengelola%20Metadata%20dengan%20Tools%20Open-Source/Latihan%20Membuat%20Version%20Control%20Menggunakan%20MLflow/train_pca.csv")

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)

# Menyimpan sample input
input_example = X_train[0:5]

# Fungsi untuk melatih,mencatat dan menyimpan Model
with mlflow.start_run():
    # Log parameters
    n_estimators = 505
    max_depth = 37
    mlflow.autolog()

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)