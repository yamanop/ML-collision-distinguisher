from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
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
df=df.sample(n=100000, random_state=42)
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)


# Hyperparameter grid define karo
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'criterion': ['gini', 'entropy']
}

# GridSearchCV object
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,  # 3-fold cross validation
    scoring='accuracy',
    n_jobs=-1,  # Sabhi CPU cores use kare
    verbose=1
)

# Fit the grid search on training data
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Test set pe evaluate
y_pred = best_model.predict(X_test)
print("Best Hyperparameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
