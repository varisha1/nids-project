# Import the required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Build ready-to-use train/test data from the CSV files
def load_dataset(cfg):
    # Get the "data" section of the config
    d = cfg["data"]
    # Read all CSV files and stack them into one table
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in d["paths"]], ignore_index=True)
    # Strip stray spaces from column names
    df.columns = [c.strip() for c in df.columns]
    # Get the name of the label column
    label_col = d["label_column"].strip()

    # If the labels live in a separate file
    if d.get("label_path"):
        # Read the label file
        lab = pd.read_csv(d["label_path"])
        # Attach the labels to the main table as a column
        df[label_col] = lab.iloc[:, 0].values[: len(df)]

    # Build the set of values that mean "normal traffic"
    benign = {str(v).strip().lower() for v in d["benign_values"]}
    # Convert labels to binary: normal = 0, anything else (attack) = 1
    y = (~df[label_col].astype(str).str.strip().str.lower().isin(benign)).astype(np.int64).values

    # List the columns to remove (id, attack_cat, and the label itself)
    drop = [c.strip() for c in d.get("drop_columns", [])] + [label_col]
    # Drop them; what remains is the feature table X
    X = df.drop(columns=drop, errors="ignore")

    # Go through every feature column
    for c in X.columns:
        # If the column holds text
        if X[c].dtype == object:
            # Try converting it to numbers
            num = pd.to_numeric(X[c], errors="coerce")
            # If more than 95% converted successfully
            if num.notna().mean() > 0.95:
                # It was really a numeric column, so keep the numeric version
                X[c] = num
            # Otherwise it is true categorical text
            else:
                # Give each distinct word an integer code (tcp=0, udp=1, ...)
                X[c] = X[c].astype("category").cat.codes
    # Force everything numeric and turn infinity into missing values (NaN)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Remember the feature names for later
    names = list(X.columns)

    # Split 70% train / 
    # 30% test, keeping the normal/attack ratio equal in both
    Xtr, Xte, ytr, yte = train_test_split(
        X.values.astype(np.float64), y,
        test_size=d.get("test_size", 0.3), stratify=y, random_state=cfg["seed"],
    )
    # Compute each column's mean using the training data only
    means = np.nanmean(Xtr, axis=0)
    # If a column was entirely missing, use 0 as its mean
    means = np.where(np.isnan(means), 0.0, means)
    # Repeat for both train and test sets
    for A in (Xtr, Xte):
        # Find the positions of missing values
        idx = np.where(np.isnan(A))
        # Fill each gap with its column's training mean
        A[idx] = np.take(means, idx[1])

    # Fit the scaler on training data only (puts all features on a similar scale)
    scaler = StandardScaler().fit(Xtr)
    # Scale the training data and convert to float32
    Xtr = np.nan_to_num(scaler.transform(Xtr)).astype(np.float32)
    # Scale the test data using the same training-fitted scaler
    Xte = np.nan_to_num(scaler.transform(Xte)).astype(np.float32)
    # Return the prepared data
    return Xtr, Xte, ytr, yte, names