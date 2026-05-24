# README.md

## ChurnZero — Cost-Aware Customer Churn Intelligence Framework

### *Shreyas Mishra*, access here - * https://colab.research.google.com/drive/1HLsCeeLOCwgdTiXveICooPP4nDqQOHRq#scrollTo=flNeINzYmMhK *

---

# Overview

This project presents a **cost-aware, behaviorally interpretable customer churn intelligence framework** developed for the **ChurnZero ML Hack — IIT Kharagpur 2026**.

Unlike conventional churn pipelines that optimize predictive metrics in isolation, this framework approaches churn as a **progressive behavioral deterioration process** emerging through:

* digital disengagement,
* service friction,
* financial stress accumulation,
* weakening relationship depth,
* and withdrawal behavior.

The central objective is not merely predicting churn, but minimizing **expected financial loss** under asymmetric banking economics while preserving operational explainability.

---

# Core Hypothesis

> Customers rarely disappear abruptly.
> Churn manifests progressively through detectable behavioral weakening.

---

# Key Design Principles

* Leakage-safe modeling
* Behavioral feature engineering
* Cost-aware threshold optimization
* Calibrated probability estimation
* Explainable decision intelligence
* Persona-driven intervention strategy
* Validation rigor through nested cross-validation

---

# Dataset Summary

| Component     | Details                |
| ------------- | ---------------------- |
| Training Rows | ~8100                  |
| Test Rows     | ~2027                  |
| Features      | ~97                    |
| Domain        | Banking Customer Churn |
| Primary Task  | Binary Classification  |

---

# Problem Framing

Traditional churn systems frequently fail because they optimize:

* accuracy,
* ROC-AUC,
* or default thresholds,

while ignoring:

* asymmetric business economics,
* temporal leakage,
* and operational deployment realism.

The competition explicitly defines:

[
Cost = 40000FN + 500FP
]

Cost = 40000FN + 500FP

Meaning:

* missing a churner is 80x more expensive than a false alarm.

Therefore, threshold selection is treated as a business optimization problem rather than a statistical convention.

---

# Pipeline Architecture

## 1. Structural Audit & Leakage Detection

The pipeline begins with:

* schema validation,
* feature-risk assessment,
* and temporal leakage auditing.

High-risk variables such as:

* `retention_offer_accepted`,
* `account_inactive_days`,
* `last_campaign_response_days`

were removed to preserve deployment legitimacy.

---

## 2. Behavioral Feature Engineering

Raw operational variables were transformed into behavioral deterioration signals.

### Examples

| Behavioral Signal        | Interpretation                |
| ------------------------ | ----------------------------- |
| engagement_decay         | weakening digital interaction |
| complaint_friction_score | service dissatisfaction       |
| withdrawal_stress_signal | financial withdrawal behavior |
| product_stickiness       | relationship depth            |
| credit_stress_index      | financial stress accumulation |

---

## 3. Model Selection Strategy

Formal baseline comparison was conducted across:

* Logistic Regression
* Random Forest
* LightGBM
* CatBoost

Final architecture selected:

# CatBoostClassifier

Reason:

* native categorical handling,
* strong structured-data performance,
* robust probability estimation,
* reduced preprocessing distortion.

---

## 4. Validation Methodology

The framework uses:

# Nested Stratified Cross Validation

### Outer CV

* unbiased model evaluation

### Inner CV

* probability calibration only

This eliminates:

* calibration leakage,
* validation contamination,
* and inflated performance estimation.

---

# Probability Calibration

Probability estimates were calibrated using:

# Isotonic Calibration

Reason:
Retention teams act on:

* probability estimates,
* intervention priorities,
* and budget allocation.

Uncalibrated probabilities reduce operational trust.

---

# Threshold Optimization

Instead of using:

```python
threshold = 0.50
```

the framework searches thresholds minimizing expected business cost:

[
40000FN + 500FP
]

This enables economically aligned churn prediction.

---

# Explainability Layer

The framework integrates:

# SHAP Explainability

Objective:

* identify stable behavioral deterioration drivers,
* understand progression mechanics,
* and preserve intervention transparency.

Additionally:

* fold-wise feature stability analysis
  was performed to verify consistency across validation folds.

---

# Behavioral Persona Segmentation

The framework extends beyond prediction into:

# behavioral interpretation.

Customers were grouped into:

* Silent Drifters
* Financially Stressed
* Frustrated Customers
* Disengaged Multi-Product
* Stable at Risk

This enables:

* targeted retention strategy,
* differentiated intervention allocation,
* operational prioritization.

---

# Performance Philosophy

The objective is NOT:

* maximizing leaderboard metrics,
* feature abundance,
* or unnecessary model complexity.

The objective IS:

* minimizing expected financial loss,
* maintaining explainability,
* and preserving operational trustworthiness.

---

# Repository Structure

```text
├── ChurnZero_FINAL_Colab_Notebook.py
├── ChurnZero_ShreyasMishra_Predictions.csv
├── Cost-Aware_Churn_Intelligence.pdf
├── README.md
```

---

# Running the Pipeline

## Install Dependencies

```bash
pip install catboost lightgbm shap pandas scikit-learn numpy matplotlib seaborn
```

---

## Execute

```bash
python ChurnZero_FINAL_Colab_Notebook.py
```

---

# Final Insight

This framework demonstrates that:

> churn prediction succeeds not when it maximizes metrics, but when it minimizes expected financial loss while remaining strategically transparent.

The final question is therefore not:

> “Will the customer churn?”

but:

> “Why is behavioral deterioration emerging, and how should intervention be prioritized?”
