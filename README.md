# Counterfactual-based Explainable Oversampling (CEOS) for Software Defect Prediction

This repository contains the replication package and source code for the paper: **"Counterfactual-based Oversampling Approach for Explainable Software Defect Prediction"**.

## 📖 Overview

Software Defect Prediction (SDP) is crucial for identifying defect-prone modules early in the software development lifecycle. However, SDP models frequently suffer from **severe class imbalance** (defective modules are rare) and a **lack of explainability** (black-box predictions).

CEOS addresses both challenges simultaneously. It operates as a powerful data synthesis engine that generates credible, boundary-proximal counterfactual samples. These samples are utilized to:

1. **Augment training data:** Improving the predictive performance of models in both Within-Project (WPDP) and Cross-Project (CPDP) defect prediction.

2. **Enhance Explainability:** Driving two well-established explainability paradigms:
   
   * Training high-fidelity **Local Surrogate Models** (e.g., Logistic Regression).
   
   * Deriving actionable **Defect Reduction Plans (Action Rules)**, which, when coupled with **refatoring methods (e.g., refactoring.guru)** and  **Large Language Models (LLMs)**, provide concrete "How-to-fix" guidance.
     
     <img src="file:///D:/CEOS/overview.png" title="" alt="overview" data-align="inline">

---

## 📂 Repository Structure

```text
├── dataset/               # Contains 32 pre-processed datasets used in the study
│   ├── promise_cpdp/      # Data used in Cross-Project Defect Prediction 
│   ├── promise_k_test/    # Data used in K-test
│   ├── select/            # Data used in Within-Project Defect Prediction         
├── utils/                 # Utility scripts for data processing and evaluation metrics
│   ├── data_details.py    # Detailed information of the dataset
│   ├── metrics.py         # Balance, R2-score, Jaccard, Validity, Proximity, Plausibility
│   └── baseline_util.py
├── exp/                   # Experimental execution scripts for each Research Question (RQ)
│   ├── RQ1                # Scripts for parameter sensitivity analysis
│   ├── RQ2                # Scripts for ablation studies (CEOS_cs, CEOS_os, CEOS_us, and CEOS_fm)
│   ├── RQ3                # Scripts for Surrogate Models and Action Rules + A case study
│   ├── RQ4                # Within-Project Defect Prediction benchmark execution
│   └── RQ5                # Cross-Project Defect Prediction benchmark execution
├── results/               # Generated predictions, action rules, and ablation results
└── README.md
```

---

## 📊 Datasets

We comprehensively evaluated CEOS on 32 real-world software datasets spanning five public corpora, encompassing diverse prediction granularities (file, class, and commit levels).

* **PROMISE**: ant, arc, camel, ivy, jedit, log4j, lucene, poi, synapse, tomcat, velocity, xalan, xerces
* **AEEEM**: Equinox, JDT, ML, PDE
* **NASA**: KC1, MC2, MW1
* **ReLink**: apache, zxing
* **MobileApps**: Alfresco, androidSync, androidWallpaper, anySoftKeyboard, Apg, facebook, kiwis, owncloudandroid, Pageturner, reddit

---

## 🔬 Experimental Baselines & Comparisons

To rigorously validate the effectiveness of CEOS, we benchmarked it against a wide array of state-of-the-art (SOTA) methods across different paradigms:

**1. Within-Project Defect Prediction (WPDP):**

* *Oversampling:* SMOTE, Borderline-SMOTE, SMOTUNED, MAHAKIL, WACIL, MPOS, MOSIG, CFOS, CFSVM
* *Hybrid & Algorithm-level:* CBR, STr-NN, CLI
* *Deep Learning:* SBGAN, WGANGP

**2. Cross-Project Defect Prediction (CPDP):**

* TCA, TCA+, NN-filter, BDA, TCA+STr-NN, DSSDPP

**3. Explainability Baselines:**

* **LIME:** For evaluating the fidelity and stability of local surrogate models.
* **TimeLIME:** For evaluating longitudinal defect reduction planning (Action Rules).

---

## 🏆 Key Experimental Results

Our experimental evaluations (documented in the `exp/` directory) answer five critical Research Questions:

### RQ1: Parameter Sensitivity

CEOS achieves optimal balance between exploration and exploitation with default settings of Iterations ($T=5$) and Neighbors ($k=5$).

### RQ2: Ablation Study

Both the Majority-to-Minority Switching (M2MS) and the bidirectional synthesis mechanism (MMOS) contribute significantly to the final predictive capabilities. CEOS consistently outperforms its partial variants.

### RQ3: Model Explainability

1. **Local Fidelity:** CEOS generates balanced sample  pairs (counter-factual & borderline-factual) that allow local surrogate models (LR) to accurately capture the global black-box logic, outperforming LIME with an average **$R^2$-score of 0.75** (vs. 0.68) and an average **Jaccaard  similarity of 0.87** (vs. 0.83).
2. **Actionability:** The derived action rules achieve an **Overlap Score of 88.63%** with actual historical developer refactoring behaviors, vastly outperforming TimeLIME (82.15%).
3. **Counterfactual Quality:** A case study is provided in the supplementary materials for readers specifically interested in the general XAI algorithmic benchmarking. We assess the quality of generated counterfactuals through visual comparison and quantitative comparison (vs. DiCE, Alibi, and LIME).

### RQ4 & RQ5: Predictive Performance (WPDP & CPDP)

CEOS demonstrates significant statistical superiority (Wilcoxon signed-rank test, $p < 0.05$) across standard evaluation metrics (Balance, F1, AUC, G-mean) using robust classifiers (RF, MLP, KNN, LR) in both single-project and cross-project scenarios.

---

## 🚀 Getting Started

**Clone the repository and install dependencies:**

```bash
git clone https://github.com/ytyancp/CEOS.git
cd CEOS
pip install -r requirements.txt
```

**Execute Research Question specific scripts in the `exp/` folder:**

```bash
# For example, you can run the single-source CPDP benchmarking
python exp/RQ5/run_CPDP_single.py

# Similarly, you can also run the multi-source CPDP benchmarking
python exp/RQ5/run_CPDP_multi.py
```

---

## 🔗 Citation

If you find this code or our conceptual framework useful in your research, please consider citing our paper:
*(Citation details will be updated upon publication)*

## 🤝 Acknowledgments

Professor Yan (faculty at AHU) supported my work throughout this project, spanning the year from April 2025 to present (March 2026). For any inquiries, please contact the author at `2151476673@qq.com`.
