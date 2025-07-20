import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("smeared_combined.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=["event", "particle_id"])
df = df[df['label'].notna()]
df["label"] = df["label"].astype(int)
df = df.fillna(0)

X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
df=df.sample(n=500000, random_state=42)
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)

# Predictions (needed for later plotting)
y_pred = clf.predict(X_test)


#importances = clf.feature_importances_
#features = X.columns
#indices = importances.argsort()[::-1]

#plt.figure(figsize=(10,6))
#sns.barplot(x=importances[indices], y=features[indices])
#plt.title("Feature Importance")
#plt.xlabel("Importance")
#plt.ylabel("Features")
#plt.tight_layout()
#plt.savefig("feature_importance.png", dpi=300)

#from sklearn.metrics import confusion_matrix


#cm = confusion_matrix(y_test, y_pred)
#plt.figure(figsize=(6,5))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#plt.title("Confusion Matrix")
#plt.xlabel("Predicted")
#plt.ylabel("Actual")
#plt.tight_layout()
#plt.savefig("confusion_matrix.png", dpi=300)


#from sklearn.metrics import roc_curve, roc_auc_score, auc

#y_prob = clf.predict_proba(X_test)[:,1]
#fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#roc_auc = auc(fpr, tpr)

#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
##plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.legend(loc="lower right")
#plt.tight_layout()
#plt.savefig("roc_curve3.png", dpi=300)

#print("AUC Score:", roc_auc_score(y_test, y_prob))






