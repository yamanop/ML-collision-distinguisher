import pandas as pd


df_signal = pd.read_csv("higgs_data.csv")
df_qcd = pd.read_csv("QCD_data.csv")




df_signal["label"] = 1
df_qcd["label"] = 0

df_combined = pd.concat([df_signal, df_qcd])
df_combined = df_combined.sample(frac=1).reset_index(drop=True)
df_combined.to_csv("combined.csv", index=False)


