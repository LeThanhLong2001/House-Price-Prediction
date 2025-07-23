# src/train_model.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

import joblib
import os

# Đọc dữ liệu
df = pd.read_csv("data/raw/train.csv")

# Loại bỏ cột Id và xử lý các biến đầu vào
df.drop(columns=["Id"], inplace=True)

# Tách biến mục tiêu
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Xử lý biến phân loại đơn giản
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Điền missing value
X[categorical_cols] = X[categorical_cols].fillna("missing")
X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

# Label encoding cho biến phân loại
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi động MLflow tracking
mlflow.set_experiment("house-price-prediction")

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")

    # Logging thông số
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("rmse", rmse)

    # Lưu model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)

    # Đăng ký mô hình với MLflow
    mlflow.sklearn.log_model(model, artifact_path="random_forest_model")
