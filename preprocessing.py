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

    load_gen_cols = [col for col in df.columns if "load" in col or "gen" in col]
    vm_cols       = [col for col in df.columns if "vm_pu" in col]
    va_cols       = [col for col in df.columns if "va_degree" in col]
    p_bus_cols    = [col for col in df.columns if col.startswith("bus_") and col.endswith("_p_mw")]
    q_bus_cols    = [col for col in df.columns if col.startswith("bus_") and col.endswith("_q_mvar")]

    X     = df[load_gen_cols].values.astype(np.float32)
    y     = df[vm_cols + va_cols].values.astype(np.float32)
    p_bus = df[p_bus_cols].values.astype(np.float32)
    q_bus = df[q_bus_cols].values.astype(np.float32)

    return X, y, p_bus, q_bus
load_data(FILE_PATH)

def preprocess(file_path=FILE_PATH):
    X, y, p_bus, q_bus = load_data(file_path)

    idx_train, idx_temp = train_test_split(np.arange(len(X)), test_size=0.3, random_state=42)
    idx_val, idx_test   = train_test_split(idx_temp, test_size=0.5, random_state=42)

    X_train, X_val, X_test       = X[idx_train],     X[idx_val],     X[idx_test]
    y_train, y_val, y_test       = y[idx_train],     y[idx_val],     y[idx_test]
    p_train, p_val = p_bus[idx_train], p_bus[idx_val]  # p/q_test not needed: physics loss is train/val only
    q_train, q_val = q_bus[idx_train], q_bus[idx_val]

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_s = scaler_X.transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train)
    y_val_s   = scaler_y.transform(y_val)
    y_test_s  = scaler_y.transform(y_test)

    np.savez(PROCESSED_PATH,
             X_train=X_train_s, X_val=X_val_s, X_test=X_test_s,
             y_train=y_train_s, y_val=y_val_s, y_test=y_test_s,
             p_train=p_train,   p_val=p_val,
             q_train=q_train,   q_val=q_val)

    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    print(f"Saved processed arrays to {PROCESSED_PATH}")
    print(f"Saved scalers to {SCALER_X_PATH}, {SCALER_Y_PATH}")

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y


def load_processed():
    d = np.load(PROCESSED_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return (d["X_train"], d["X_val"], d["X_test"],
            d["y_train"], d["y_val"], d["y_test"],
            d["p_train"], d["p_val"],
            d["q_train"], d["q_val"],
            scaler_X, scaler_y)

if __name__ == "__main__":
    preprocess()
