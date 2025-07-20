import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
df = pd.read_csv("smeared_combined.csv", sep=None, on_bad_lines="skip")
df.columns = df.columns.str.strip()

# Fill event ID properly
if "event" in df.columns:
    df["event"] = df["event"].fillna(method="ffill")

# Drop unwanted columns
df = df.drop(columns=["particle_id"], errors='ignore')

# Ensure label column is clean
df = df[df['label'].notna()]
df["label"] = df["label"].astype(int)
df = df.fillna(0)

# -------------------------------
# ✅ Feature Engineering Part
# -------------------------------

# Composite Features
df["pt_eta"] = df["pt"] * df["eta"]
df["pt_phi"] = df["pt"] * df["phi"]
df["mass_pt"] = df["mass"] * df["pt"]
df["pt_div_eta"] = df["pt"] / (df["eta"] + 1e-5)
df["mass_div_phi"] = df["mass"] / (df["phi"] + 1e-5)

# Log Transform Features
df["log_pt"] = np.log1p(df["pt"])
df["log_mass"] = np.log1p(df["mass"])

# Square Terms
df["eta2"] = df["eta"]**2
df["phi2"] = df["phi"]**2

# Sum of pt in same event
df["sum_pt_event"] = df.groupby("event")["pt"].transform("sum")
df["pt_ratio"] = df["pt"] / (df["sum_pt_event"] + 1e-5)

# Drop event if no longer needed
df = df.drop(columns=["event"], errors='ignore')

# -------------------------------
# ✅ Machine Learning Training
# -------------------------------

# Separate features and label
X = df.drop(["label"], axis=1)
y = df["label"]

# Sample if needed (for speed)
df = df.sample(n=500000, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest model
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
