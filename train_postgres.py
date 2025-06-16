import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# --- DATABASE CONFIGURATION ---
DB_USER = "your_user"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "your_db"
TABLE_NAME = "camera_events"


# --- CONNECT TO POSTGRESQL AND LOAD DATA ---
def load_data_from_postgres():
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(connection_string)
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql(query, engine)
    return df


if __name__ == "__main__":
    from preprocessing.clean_and_merge import clean_camera_data
    from preprocessing.feature_engineering import add_time_features, encode_event_type

    # 1. Load data from PostgreSQL
    df = load_data_from_postgres()
    print(f"Loaded {len(df)} rows from PostgreSQL.")

    # 2. Clean and preprocess
    df = clean_camera_data(df)
    df = add_time_features(df)
    df = encode_event_type(df)

    # 3. Assume 'anomaly_flag' is the target
    if "anomaly_flag" not in df.columns:
        raise ValueError("'anomaly_flag' column not found in the database table.")
    X = df.drop(columns=["anomaly_flag"]).values
    y = df["anomaly_flag"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 4. Train Random Forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 5. Save the model
    os.makedirs("../models", exist_ok=True)
    with open("../models/rf_model_postgres.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved as ../models/rf_model_postgres.pkl")
