import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load & clean data
df = pd.read_csv("smeared_combined.csv", sep=None, on_bad_lines="skip")
df.columns = df.columns.str.strip()
df = df.drop(columns=["event", "particle_id"], errors='ignore')
df = df[df['label'].notna()]
df["label"] = df["label"].astype(int)
df = df.fillna(0)

# Split
X = df.drop(["label"], axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gradient Boosting
df=df.sample(n=500000, random_state=42)
gboost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gboost.fit(X_train, y_train)

# Predict
y_pred_gb = gboost.predict(X_test)

# Model train karna (agar already trained nahi hai)
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

# Feature importance lena
importances = gb.feature_importances_
features = X_train.columns

# Sort karna
indices = importances.argsort()[::-1]

# Plot banana
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances - Gradient Boosting")
plt.xlabel("Importance")
plt.ylabel("Features")

# Save karna image
plt.tight_layout()
plt.savefig("gb_feature_importance.png", dpi=300)
plt.close()

print(" Gradient Boosting feature importance plot saved as 'gb_feature_importance.png'")
