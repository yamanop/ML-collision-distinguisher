import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Data
df = pd.read_csv("smeared_combined.csv", sep=None, engine='python', on_bad_lines="skip")
df.columns = df.columns.str.strip()
df = df.drop(columns=["event", "particle_id"])
df = df[df['label'].notna()]
df["label"] = df["label"].astype(int)
df = df.fillna(0)

# Step 2: Features & Labels
X = df.drop(["label"], axis=1)
y = df["label"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df=df.sample(n=50000, random_state=42)

# Step 4: Define Models
models = {

    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Step 5: Cross-Validation
print("===== Cross Validation Results (5-Fold) =====")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Step 6: Train Best Model (you can choose later based on scores)
best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Step 7: Final Evaluation
print("\n===== Final Evaluation on Test Set =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
