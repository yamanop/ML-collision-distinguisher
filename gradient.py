import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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




# Evaluation
#print(" Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
#print("\n Classification Report:\n", classification_report(y_test, y_pred_gb))
#print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))



from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

y_prob = gboost.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve4.png", dpi=300)

print("AUC Score:", roc_auc_score(y_test, y_prob))


