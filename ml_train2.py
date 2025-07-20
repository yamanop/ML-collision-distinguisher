import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the data
df = pd.read_csv("smeared_combined.csv", sep=None, on_bad_lines="skip")
df.columns = df.columns.str.strip()

df = df.drop(columns=["event", "particle_id"])

df = df[df['label'].notna()]
df["label"] = df["label"].astype(int)
df = df.fillna(0)



# Optional: drop non-numeric or irrelevant columns
if "event" in df.columns:
    df = df.drop(columns=["event"])


# Split into features and labels
X = df.drop(["label"], axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
df=df.sample(n=500000, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
