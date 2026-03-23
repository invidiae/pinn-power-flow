import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FILE_PATH = "data/ieee_14.csv"
PROCESSED_PATH = "data/processed.npz"
SCALER_X_PATH = "data/scaler_X.pkl"
SCALER_Y_PATH = "data/scaler_Y.pkl"


def load_data(file_path):
    df = pd.read_csv(file_path)

    load_cols = [col for col in df.columns if "load" in col]
    bus_cols = [col for col in df.columns if "bus" in col]

    X = df[load_cols].values.astype(np.float32)
    y = df[bus_cols].values.astype(np.float32)

    return X, y


def preprocess(file_path=FILE_PATH):
    X, y = load_data(file_path)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_s = scaler_X.transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    np.savez(PROCESSED_PATH,
             X_train=X_train_s, X_val=X_val_s, X_test=X_test_s,
             y_train=y_train_s, y_val=y_val_s, y_test=y_test_s)

    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    print(f"Saved processed arrays to {PROCESSED_PATH}")
    print(f"Saved scalers to {SCALER_X_PATH}, {SCALER_Y_PATH}")

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y


def load_processed():
    d = np.load(PROCESSED_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return d["X_train"], d["X_val"], d["X_test"], d["y_train"], d["y_val"], d["y_test"], scaler_X, scaler_y

if __name__ == "__main__":
    preprocess()
