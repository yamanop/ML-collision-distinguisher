import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv("smeared_combined.csv", sep=None, on_bad_lines="skip")
df.columns = df.columns.str.strip()
df = df.drop(columns=["event", "particle_id"], errors='ignore')
df = df[df['label'].notna()]
df["label"] = df["label"].astype(int)
df = df.fillna(0)

X = df.drop(["label"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)

# SHAP
# Subset for faster SHAP
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample subset
X_small = X_train.sample(n=1000, random_state=42)

# SHAP Explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_small)

# Convert to NumPy array if needed (avoid "truth value" or shape mismatch errors)
if isinstance(shap_values, list):
    shap_val = np.array(shap_values[1])  # class 1
else:
    shap_val = np.array(shap_values)

X_small_np = X_small.to_numpy()

# Top 3 features
shap_abs_mean = np.abs(shap_val).mean(axis=0)
top_indices = np.argsort(shap_abs_mean)[-3:][::-1]
top_features = X_small.columns[top_indices]

print("Top SHAP features:", top_features)

# Dependence plots
for feature in top_features:
    plt.figure()
    shap.dependence_plot(
        feature,
        shap_val,
        X_small,
        show=False,
        interaction_index=None  # Avoids weird array shape issues
    )
    plt.title(f"SHAP Dependence Plot - {feature}")
    plt.savefig(f"shap_dependence_{feature}.png", dpi=300, bbox_inches='tight')
    plt.close()
