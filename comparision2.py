import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have already trained these two:
models = {
    "Random Forest": clf,         # Already trained
    "Gradient Boosting": gboost  # Already trained
}

# Dictionary to store results
results = []

# Calculate metrics for each model
for name, model in models.items():
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-Score": round(f1, 3)
    })

# Create DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Save to CSV
results_df.to_csv("model_comparison_results.csv", index=False)
print("Comparison saved to model_comparison_results.csv")

# Plot as image table
fig, ax = plt.subplots(figsize=(8, 2))  # Resize as needed
ax.axis('off')
table = ax.table(cellText=results_df.values,
                 colLabels=results_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

plt.title("Model Performance Comparison", fontsize=14, weight='bold')
plt.savefig("model_comparison_table.png", dpi=300, bbox_inches='tight')
plt.close()

print(" Table image saved as model_comparison_table.png")
