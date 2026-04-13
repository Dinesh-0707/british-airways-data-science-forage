# Customer Sentiment Analysis for Airline Reviews

![British Airways](https://img.shields.io/badge/British%20Airways-Data%20Science-002157?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## Overview

This repository contains my completed work for the **British Airways Data Science Job Simulation** hosted on [Forage](https://www.theforage.com/simulations/british-airways/data-science-yqoz).

The simulation placed me in the role of a data scientist on British Airways' Data Science team, working on two real-world business problems: modelling lounge demand at Heathrow Terminal 3, and predicting customer booking behaviour using machine learning.

- Problem :
Airlines receive thousands of reviews — hard to extract insights.

- Solution:
Built an NLP pipeline to classify sentiment and identify key issues.

- Tools:
Python, Pandas, NLTK, Scikit-learn

- Key Insights: 
Negative sentiment linked to delays and staff behaviour
Positive sentiment driven by service quality

- Business Impact: 
Helps airlines improve customer satisfaction and retention.
---

## Repository Structure

```
british-airways-data-science-forage/
│
├── Task1/
│   └── BA_Task1_Lounge_Eligibility_Dinesh.xlsx   # Lounge eligibility lookup table + justification
│
├── Task2/
│   └── BA_Task2_Dinesh.ipynb                     # Customer booking prediction model
│
├── Presentation/
│   └── BA_Task2_Summary_Dinesh.pptx              # Manager summary slide deck
│
├── outputs/
│   ├── ba_eda_overview.png                        # EDA: 6-panel booking behaviour analysis
│   ├── ba_feature_importance.png                  # Random Forest feature importance chart
│   ├── ba_cv_metrics.png                          # 5-fold cross-validation metrics
│   └── ba_roc_curve.png                           # ROC curve (AUC = 0.765)
│
└── README.md
```

---

## Task 1 — Lounge Demand Modelling at Heathrow Terminal 3

**Objective:** Create a reusable lookup table that British Airways can use to estimate lounge eligibility percentages across different flight groupings — without needing specific aircraft or passenger manifest data.

### Context
BA's Heathrow Terminal 3 has three lounge tiers:
| Tier | Lounge | Eligibility |
|------|--------|-------------|
| 1 | Concorde Room | First Class, BA Premier cardholders, BA Gold Guest List |
| 2 | First Lounge | BA Gold Members |
| 3 | Club Lounge | BA Silver cardholders, Club World (business class) customers |

### Approach
- Analysed a **10,000-flight summer schedule dataset** to derive data-driven eligibility percentages
- Grouped flights by **Haul Type** (Long-Haul / Short-Haul) × **Time of Day** (Morning / Lunchtime / Afternoon / Evening) — 8 categories total
- Built a secondary **regional cross-reference** table (Europe / North America / Middle East / Asia)
- Applied the lookup table to a **sample of real flights** to validate demand estimates

### Key Results
| Group | Tier 1 % | Tier 2 % | Tier 3 % |
|-------|----------|----------|----------|
| Long-Haul Morning | 0.2% | 3.0% | 11.3% |
| Long-Haul Evening | 0.2% | 2.9% | 11.0% |
| Short-Haul Morning | 0.4% | 4.4% | 16.7% |
| Short-Haul Evening | 0.3% | 4.5% | 17.0% |

### Deliverable
- **Excel file** with two sheets: Lookup Table (with applied sample) + Justification (4 reflective questions answered)

---

## Task 2 — Predicting Customer Booking Behaviour

**Objective:** Train a machine learning model to predict whether a customer will complete a booking, and interpret which variables are most predictive.

### Dataset
- 50,000 customer booking records
- 13 input features (numeric and categorical)
- Target: `booking_complete` (1 = completed, 0 = not completed)
- Class imbalance: **85% non-complete / 15% complete**

### Feature Engineering
10 new features created on top of the original 13:

| Feature | Description |
|---------|-------------|
| `total_addons` | Sum of extra baggage + preferred seat + in-flight meals |
| `is_mobile` | Flag: booked via mobile app |
| `is_round_trip` | Flag: round trip booking |
| `is_early_flight` | Flag: departure before 06:00 |
| `is_weekend` | Flag: Saturday or Sunday departure |
| `is_last_minute` | Flag: booked within 7 days of flight |
| `is_planned_ahead` | Flag: booked more than 90 days ahead |
| `flight_day_num` | Ordinal encoding of flight day (Mon=0 to Sun=6) |
| `route_freq` | How frequently this route appears in the dataset |
| `origin_freq` | How frequently this booking origin appears |

### Model
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:** 500 trees, max depth 12, min samples per leaf 10
- **Class imbalance handling:** `class_weight='balanced'`
- **Validation:** 5-fold Stratified Cross-Validation

### Results

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.765** ← primary metric |
| Accuracy | 72.3% |
| Recall | 66.3% |
| F1 Score | 0.417 |
| Precision | 30.4% |

### Feature Importance

![Feature Importance](outputs/ba_feature_importance.png)

**Top predictors:**
1. **Booking Origin** (37% importance) — where the customer books from is the strongest signal
2. **Route Frequency** (12%) — more popular routes show distinct booking completion patterns
3. **Length of Stay** (11%) — planned holidays with specific durations show higher completion

### EDA Highlights

![EDA Overview](outputs/ba_eda_overview.png)

- Mobile bookings complete at **22% vs 13%** for Internet — mobile users show stronger purchase intent
- Customers selecting all 3 add-ons complete at **3× the rate** of zero add-on customers
- Last-minute bookings (≤7 days) show significantly **lower** completion rates

### Commercial Recommendation
BA should deploy **geo-targeted proactive marketing** — identifying high-propensity customers by origin market and route before they leave the browse stage. Customers who select add-ons show strong purchase intent and should be prioritised for re-engagement campaigns.

---

## Skills Demonstrated

- **Feature engineering** — creating 10 new predictive features from raw data
- **Class imbalance handling** — `class_weight='balanced'`, ROC-AUC as primary metric
- **Model evaluation** — 5-fold stratified cross-validation, multiple metrics
- **Interpretability** — Random Forest feature importance, commercial insight translation
- **Data visualisation** — EDA, feature importance charts, ROC curve
- **Business communication** — single-slide manager summary, actionable recommendations

## Certificate

Completed via the [British Airways Data Science Job Simulation on Forage](https://www.theforage.com/simulations/british-airways/data-science-yqoz).

---

*Part of an ongoing data analytics portfolio. Connect on [LinkedIn](https://linkedin.com/in/dineshnavaneethan).*
