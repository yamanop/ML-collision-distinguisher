# Higgs vs QCD Classification using Simulation and Machine Learning

This repository contains the full pipeline for simulating particle collisions using **Pythia8**, applying **realistic smearing**, training **machine learning models** (Random Forest and Gradient Boosting), and visualizing the results. The goal is to classify Higgs boson events from QCD background using ML, as a part of a research-oriented physics-CS integration project.

---

## 📌 Project Workflow

Pythia8 (Higgs & QCD event generation)
↓
CSV Conversion (event-wise storage)
↓
Smearing (Detector realism: pT, η, ϕ, mass)
↓
ML Pipeline (Feature Engineering + Training)
↓
Evaluation & Visualization (Confusion Matrix, ROC, SHAP)


---


## 🚀 Features

- ✅ Pythia8-based Higgs and QCD event generation (C++)
- ✅ Detector smearing applied for realism
- ✅ Data conversion to CSV for ML usage
- ✅ Random Forest & Gradient Boosting classifiers
- ✅ Feature Engineering & Model Comparison
- ✅ SHAP Explainability, ROC curves, Confusion Matrix plots
- ✅ Final accuracy ~79% with visual insights



---

##  🧪 Requirements

Run inside WSL or Linux-based system or install pythia8 and run all the files inside ~/pythia8-install/share/Pythia8/examples  folder.


Thanks to the CERN Open Data initiative and Pythia8 authors for enabling reproducible and research-level simulation.


For queries or collaboration, reach out at:
📩 [yaman1857.360@gmail.com]
