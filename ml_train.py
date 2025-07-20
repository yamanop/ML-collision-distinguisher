import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("higgs_dataset.csv", sep="|")  # <-- IF your file uses |

# Clean and prepare
X = df.drop(columns=["label"], errors="ignore")
y = pd.Series([1]*len(X))  # Sab ko 1 (signal) maana hai abhi ke lie


# Convert to numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
