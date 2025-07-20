import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load combined data
df = pd.read_csv("combined.csv", sep=None, on_bad_lines="skip")

# Drop missing or irrelevant columns
df = df.fillna(0)
if "event" in df.columns:
    df = df.drop(columns=["event"])

# Plot pT distributions
plt.figure(figsize=(10, 6))
sns.histplot(df[df["label"] == 1]["pt"], bins=100, color='red', label='Higgs', kde=True)
sns.histplot(df[df["label"] == 0]["pt"], bins=100, color='blue', label='QCD', kde=True)
plt.xlabel("Transverse Momentum (pt)")
plt.ylabel("Frequency")
plt.title("pt Distribution: Higgs vs QCD")
plt.legend()
plt.tight_layout()
plt.savefig("pt_distribution.png")




