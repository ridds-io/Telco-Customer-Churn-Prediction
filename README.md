# Telco Customer Churn Prediction — Kaggle Playground Series S6E3

![Python](https://img.shields.io/badge/Python-3.10-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-✓-green) ![PyTorch](https://img.shields.io/badge/PyTorch-GNN-orange)

**Competition:** [Playground Series — Season 6 Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3/overview)  
**Task:** Binary classification — predict customer churn (Yes/No)  
**Metric:** ROC-AUC  
**Final Score:** 0.91638 | **Final Rank:** 790 / 4142  

<img width="1600" height="114" alt="image" src="https://github.com/user-attachments/assets/434c6440-1a7b-4a54-b700-2eca62bc61c3" />

**Peak Rank (mid-competition):** 353 / 1200+

<img width="1280" height="184" alt="image" src="https://github.com/user-attachments/assets/5cc22428-2dd6-4f1e-a202-226159670b5b" />


---

## Overview

End-to-end machine learning pipeline on 594,000 rows of synthetic telecom data. The work covers the full competition arc: initial data understanding, a wide range of modeling and feature engineering experiments (most of which failed usefully), and a final breakthrough from a Graph Neural Network that exploited the synthetic structure of the dataset.

This README documents what was tried, what worked, and — just as importantly — what didn't and why.

---

## The Key Insight

The competition dataset was synthetically generated from ~7,000 original IBM Telco rows. Each real customer was copied approximately **85 times** with added noise. This creates a structure that a Graph Neural Network can exploit: build a KNN graph on the feature space, and each node's nearest neighbours are almost certainly its copies. If a customer's copies are mostly churners, the model gains very high confidence about that customer's label.

This single observation — discovered from reading competition discussion posts — led to the GraphSAGE model that became the biggest single score improvement in the whole competition.

---

## Notebook Contents

| # | Section | What's Covered |
|---|---|---|
| 1 | Setup | Imports, environment, GPU (cuML/CUDA) |
| 2 | Data Loading | 594k train rows + IBM supplementary data |
| 3 | EDA | Class balance, categorical churn rates, numeric distributions, correlation heatmap |
| 4 | Advanced EDA | Mutual information, UMAP, MDS, distribution shift analysis (train vs test), feature interaction breakdowns |
| 5 | Preprocessing | Target encoding, binary/one-hot encoding, column cleanup |
| 6 | Feature Engineering | Add-on service counts, tenure groups + dead ends (charge_per_addon, contract-payment combos, log/sqrt transforms) |
| 7 | Modeling: Logistic Regression | Baseline — AUC 0.9084, Recall 0.66 |
| 8 | Modeling: XGBoost | scale_pos_weight for imbalance, feature importance analysis, max_depth=1 stumps experiment |
| 9 | Hyperparameter Tuning | Optuna with 30 and 100 trials; GPU-accelerated search |
| 10 | Modeling: LightGBM | Two variants, learning rate sensitivity, num_leaves tuning |
| 11 | Modeling: CatBoost | Native categorical handling, GPU training |
| 12 | Things That Didn't Work | SMOTE-ENN, target encoding, KMeans clustering (k=4/5/6), Cox hazard model, pseudo-labelling, real/synthetic classifier, categorical interaction features, sample weighting by tenure |
| 13 | Neural Networks | TabNet, PyTorch Tabular (CategoryEmbeddingModel, FTTransformer), custom MLP — ensemble blends tested |
| 14 | Ensembling | Simple averaging, weight search (manual + Optuna), rank averaging, stacking, multi-seed blending |
| 15 | GNN (Breakthrough) | GraphSAGE on KNN graph via PyTorch Geometric; GPU NearestNeighbors via cuML |
| 16 | Hill Climbing | Greedy ensemble weight search that found the optimal GNN + XGBoost + CatBoost blend |
| 17 | Submission Pipeline | Full test preprocessing and prediction pipeline |
| 18 | Results & Reflections | What worked, what didn't, and what I'd do differently |

---

## EDA Findings

**Class Balance:** 77.5% No Churn / 22.5% Churn — moderately imbalanced.

**Strongest predictors (from EDA + mutual information):**

| Feature | Observation |
|---|---|
| `Contract` | Month-to-month customers churn at ~42% vs 6% (one-year) and 2% (two-year) |
| `PaymentMethod` | Electronic check → ~49% churn rate, nearly 5x higher than other methods |
| `InternetService` | Fiber optic → ~42% churn rate |
| `tenure` | Short-tenure customers churn far more; long-tenure customers are highly stable |
| `TotalCharges` | Strong inverse relationship with churn |

**Distribution shift:** The test set overrepresents `tenure=72` customers relative to training — a sign the datasets were sampled differently, explaining the val/LB gap throughout competition.

**UMAP and MDS** confirmed two distinct clusters: a tight blob of low-churn customers and a scattered region of churners, consistent with the two-signal structure later confirmed by the GNN insight.

---

## Modeling

### Logistic Regression (Baseline)
Standard scaled, `max_iter=1000`. Establishes a strong baseline at AUC 0.9084 — the linear relationships in this dataset are genuinely informative.

### XGBoost
`scale_pos_weight=3` to handle class imbalance. Churn recall jumped from 0.66 → 0.86 vs. logistic regression, with a small precision tradeoff. Feature importance confirmed that three features (`PaymentMethod_Electronic_check`, `InternetService_Fiber_optic`, `Contract_Two_year`) account for ~69% of total importance.

**max_depth=1 experiment:** After identifying the two-signal framework, trained XGBoost with max_depth=1 (decision stumps). This captures the simple, near-linear real signal from the original IBM rows cleanly. Tested at up to 20,000 estimators with a low learning rate.

**Optuna:** Two rounds — 30 trials and 100 trials on GPU. Outcome: negligible improvement over manual params (+0.0002 AUC). Optuna preferred shallower, faster-learning configurations; the model was already well-tuned manually.

### LightGBM
Tested two configurations. The first (lr=0.05, max_depth=6) hit early stopping at tree 38 — too fast. The second (lr=0.01, num_leaves=63) improved to 0.9130 but didn't surpass XGBoost. LightGBM did not outperform XGBoost on this dataset.

### CatBoost
Native categorical feature handling (no encoding needed — separate preprocessing pipeline). Trained on GPU. Matched XGBoost at ~0.9165 with a different error profile, making it a strong ensemble partner.

---

## What Didn't Work (and Why)

These experiments weren't wasted — each one confirmed something about the structure of the dataset.

**SMOTE-ENN:** Hurt performance. XGBoost's `scale_pos_weight` already handles imbalance correctly; SMOTE creates synthetic minority samples that don't match the real distribution.

**Target encoding:** No improvement. Feature independence in the synthetic data means target statistics carry no extra signal over raw categorical values.

**KMeans clustering (k=4, 5, 6):** The clusters were visually distinct in UMAP (k=4 separated a very loyal cluster with 1.4% churn and a high-risk cluster at 46.7% churn), but adding cluster labels as a model feature made no difference — XGBoost had already learned this structure from the original features.

**Cox Proportional Hazard model:** Generated a hazard score per customer using tenure as the duration and Churn as the event. Informative on its own (correlation 0.35 with Churn), but adding it as a feature produced no AUC gain — the model already captured this relationship.

**Pseudo-labelling:** Used high-confidence test predictions (≥95% or ≤5%) as training labels. No improvement. The model was already well-calibrated, and the pseudo-labels added noise rather than signal.

**Real/synthetic classifier:** Trained a classifier to distinguish IBM original rows from synthetic competition rows, then used the "probability of being real" as a feature. No AUC improvement — the model couldn't benefit from this signal given the feature space.

**Sample weighting by tenure:** Test set has fewer `tenure=1` customers than train. Downweighted tenure=1 rows by their proportion difference. Small negative effect — not worth it.

**Neural networks (TabNet, FTTransformer, custom MLP):** AUC below XGBoost individually, marginal contribution in ensembles. The dataset's feature independence means deep networks don't benefit from the cross-feature interaction learning they're built for.

---

## Ensembling

### Multi-seed blending
Same model architecture, 5 different random seeds (42, 123, 456, 789, 1024). Seeds averaged for XGBoost, CatBoost, and LightGBM. Cheap variance reduction that consistently improved AUC by ~0.0002–0.0003.

### Rank averaging
Converting raw probabilities to percentile ranks before averaging. More robust when model score distributions differ significantly. Tested alongside probability averaging.

### Stacking
Used OOF predictions from XGBoost and CatBoost as meta-features for a Logistic Regression meta-learner. Performed slightly worse than direct blending.

### Optuna ensemble search
Simultaneously optimized weights across all models. Useful for exploring the weight space but prone to overfitting on the validation set with too many models.

### Hill climbing
Greedy iterative approach: start with the best single model, then repeatedly add weight to whichever model improves the ensemble AUC most. More reliable than Optuna for ensemble search — discrete, interpretable, and less prone to overfit. This found the final GNN + XGBoost + CatBoost blend.

---

## Graph Neural Network (GraphSAGE)

The breakthrough model. Built a KNN graph (k≈85, matching the estimated copy count per original row) over all training rows using GPU-accelerated NearestNeighbors from cuML. GraphSAGE aggregates neighbourhood information — in this case, neighbourhood = copies of the same original IBM customer.

**Why this works:** If a real customer churned, ~85 synthetic copies of them are also labelled as churners. A GNN that can find those copies gains very high-confidence predictions, especially for customers near the boundary.

| Submission | Val AUC | Public LB | Notes |
|---|---|---|---|
| XGBoost + CatBoost (50/50) | ~0.9165 | 0.91393 | Pre-GNN best |
| + GNN (hill climbing blend) | ~0.9177 | 0.91638 | Final submission |

The GNN alone: +0.00125 LB score, ~464 rank positions.

---

## Results

| Model | Val AUC | Public LB |
|---|---|---|
| Logistic Regression (baseline) | 0.9084 | — |
| XGBoost (manual params) | 0.9165 | ~0.914 |
| XGBoost + Optuna (30 trials) | 0.9167 | — |
| XGBoost + CatBoost ensemble | ~0.9165 | 0.91393 |
| Multi-seed XGBoost avg | 0.9168 | — |
| + GNN (hill climbing blend) | 0.9177 | **0.91638** |

**Final rank:** 790 / 4142 | **Peak rank:** 353 / 1200+


<img width="1256" height="626" alt="image" src="https://github.com/user-attachments/assets/8431ea48-b221-49e2-abf6-837c42a87cfb" />
Submission history — public LB scores across all iterations.

---

## What I'd Do Differently

1. Use 5-fold stratified cross-validation from the start. Single val split gave noisy signals that made it hard to know when a technique genuinely helped vs. got lucky.
2. Start with data understanding before tuning. Multiple Optuna rounds produced negligible gains. That time would have been better spent understanding the dataset structure earlier.
3. Explore the original IBM dataset more deeply. Since the synthetic data was generated from ~7k IBM rows, engineering features that reflect patterns in the original — customer lifetime value proxies, service bundling behaviour — could have added signal the model wasn't already seeing.
4. Build a more systematic ensembling stage. I had TabNet, FTTransformer, a custom MLP, a Cox hazard score, cluster labels, and multiple tree-based models — but never properly combined everything into one structured ensemble search. Hill climbing at the end only covered a subset.
5. Read competition discussions earlier. The GNN insight arrived late — earlier awareness of the synthetic copy structure would have shaped the whole approach from the beginning.

---

## Running the Notebook

Designed for Kaggle with GPU enabled (P100 or T4).

```bash
pip install torch-geometric optuna lifelines pytorch-tabnet pytorch-tabular cuml-cu12
```

For CPU-only environments: remove `device='cuda'` and `cuml` imports, replace GPU-accelerated KNN with `sklearn.neighbors.NearestNeighbors`.

---

## Dependencies

Beyond the standard Kaggle Python environment: `torch-geometric`, `optuna`, `lifelines`, `pytorch-tabnet`, `pytorch-tabular`, `cuml` (GPU).

---

**Competition:** [Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)
