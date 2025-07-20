import numpy as np
import pandas as pd

df = pd.read_csv("combined.csv", sep="|")

# Define resolutions (you can tweak)
resolutions = {
    'pT': 0.10,     # 10%
    'E': 0.10,      # 10%
    'eta': 0.01,    # small
    'phi': 0.01,    # small
    'mass': 0.05    # 5%
}

for col, res in resolutions.items():
    if col in df.columns:
        df[col] = df[col] + np.random.normal(0, res * df[col])

df.to_csv("smeared_combined.csv", sep="|", index=False)
