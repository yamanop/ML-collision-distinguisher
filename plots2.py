import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("smeared_combined.csv")
df.columns = df.columns.str.strip()
df = df[df['label'].notna()]
df['label'] = df['label'].astype(int)

signal=df[df['label']==1]
background=df[df['label']==0]


#For pT- Transverse Momentum



#For eta
#plt.hist(signal['eta'], bins=100, alpha=0.6, label= 'Signal (Higgs)', color='red', density=True)
#plt.hist(background['eta'],bins=100, alpha=0.6, label='Backrgound (QCD)', color='blue', density=True)
#plt.xlabel("Pseudorapidity (eta)")
#plt.ylabel("Normalised Count")
#plt.title("Histogram of eta - Signal vs Background")
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.savefig("eta_histogram2.png",dpi=300)


#FOR MASS 
#plt.figure(figsize=(10,6))

#plt.hist(signal['mass'].dropna, bins=100, alpha=0.6, label='Signal (Higgs)', color='green', density=True)
#plt.hist(background['mass'].dropna(), bins=100, alpha=0.6, label='Background (QCD)', color='orange', density=True)

#plt.xlabel("Mass (GeV)")
#plt.ylabel("Density")
#plt.title("Mass Distribution: Higgs vs QCD")
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.savefig("mass_histogram2.png", dpi=300)


#for phi
#plt.figure(figsize=(10,6))
#plt.hist(signal['phi'], bins=100, alpha=0.6, label='Signal (Higgs)', color='cyan', density=True)
#plt.hist(background['phi'], bins=100, alpha=0.6, label='Background (QCD)', color='magenta', density=True)

#plt.xlabel('Azimuthal Angle (ϕ)')
#plt.ylabel('Normalized Count')
#plt.title('Histogram of ϕ - Signal vs Background')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.savefig("phi_histogram.png", dpi=300)


import seaborn as sns

subset = df.sample(100000)  # Take small sample for fast plotting
sns.pairplot(subset, hue="label", vars=["pt", "eta", "phi", "mass"])
plt.savefig("pairplot.png", dpi=300)












