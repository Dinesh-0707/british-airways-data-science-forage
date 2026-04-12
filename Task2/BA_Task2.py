# ============================================================
# BRITISH AIRWAYS DATA SCIENCE — TASK 2
# Predicting Customer Booking Completion
# Analyst: Dinesh Navaneethan
# ============================================================

# ── CELL 1: Imports ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 120, 'font.family': 'Arial'})

print("Libraries loaded.")


# ── CELL 2: Load Data ────────────────────────────────────────
df = pd.read_csv("customer_booking.csv", encoding="latin1")
print("Shape:", df.shape)
print("\nTarget distribution:")
print(df['booking_complete'].value_counts())
print(f"\nClass balance: {df['booking_complete'].mean()*100:.1f}% completed bookings")


# ── CELL 3: EDA ──────────────────────────────────────────────
print("=== BASIC STATISTICS ===")
print(df.describe().round(2).to_string())

print("\n=== NULL VALUES ===")
print(df.isnull().sum())

print("\n=== CATEGORICAL COLUMNS ===")
for col in ['sales_channel', 'trip_type', 'flight_day']:
    print(f"\n{col}:")
    print(df[col].value_counts())


# ── CELL 4: EDA Visualisations ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Booking completion rate by sales channel
channel_rate = df.groupby('sales_channel')['booking_complete'].mean() * 100
axes[0].bar(channel_rate.index, channel_rate.values,
            color=['#003087', '#C8102E'])
axes[0].set_title('Booking Rate by Sales Channel', fontweight='bold')
axes[0].set_ylabel('Completion Rate (%)')
axes[0].set_ylim(0, 30)
for i, v in enumerate(channel_rate.values):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Purchase lead time distribution
axes[1].hist(df[df['booking_complete']==0]['purchase_lead'],
             bins=50, alpha=0.6, color='#AAAAAA', label='Not completed', density=True)
axes[1].hist(df[df['booking_complete']==1]['purchase_lead'],
             bins=50, alpha=0.7, color='#003087', label='Completed', density=True)
axes[1].set_title('Purchase Lead Time Distribution', fontweight='bold')
axes[1].set_xlabel('Days before flight')
axes[1].set_ylabel('Density')
axes[1].legend()

# Flight duration vs booking
axes[2].hist(df[df['booking_complete']==0]['flight_duration'],
             bins=30, alpha=0.6, color='#AAAAAA', label='Not completed', density=True)
axes[2].hist(df[df['booking_complete']==1]['flight_duration'],
             bins=30, alpha=0.7, color='#003087', label='Completed', density=True)
axes[2].set_title('Flight Duration Distribution', fontweight='bold')
axes[2].set_xlabel('Duration (hours)')
axes[2].set_ylabel('Density')
axes[2].legend()

# Booking rate by trip type
trip_rate = df.groupby('trip_type')['booking_complete'].mean() * 100
axes[3].bar(trip_rate.index, trip_rate.values,
            color=['#003087', '#C8102E', '#D4AF37'])
axes[3].set_title('Booking Rate by Trip Type', fontweight='bold')
axes[3].set_ylabel('Completion Rate (%)')
for i, v in enumerate(trip_rate.values):
    axes[3].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Add-ons correlation with booking
addons = ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
addon_rates = [df[df[a]==1]['booking_complete'].mean()*100 for a in addons]
addon_labels = ['Extra Baggage', 'Preferred Seat', 'In-flight Meals']
axes[4].barh(addon_labels, addon_rates, color=['#003087', '#003087', '#003087'])
axes[4].set_title('Booking Rate for Add-on Requesters', fontweight='bold')
axes[4].set_xlabel('Completion Rate (%)')
for i, v in enumerate(addon_rates):
    axes[4].text(v + 0.1, i, f'{v:.1f}%', va='center', fontweight='bold')

# Booking rate by flight day
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_rate = df.groupby('flight_day')['booking_complete'].mean().reindex(day_order) * 100
axes[5].bar(day_rate.index, day_rate.values, color='#003087')
axes[5].set_title('Booking Rate by Flight Day', fontweight='bold')
axes[5].set_ylabel('Completion Rate (%)')
axes[5].tick_params(axis='x', rotation=0)

plt.suptitle('British Airways — Customer Booking EDA', fontsize=14,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('ba_eda_overview.png', bbox_inches='tight')
plt.show()


# ── CELL 5: Feature Engineering ──────────────────────────────
df_model = df.copy()

# 1. Total add-ons requested
df_model['total_addons'] = (df_model['wants_extra_baggage'] +
                             df_model['wants_preferred_seat'] +
                             df_model['wants_in_flight_meals'])

# 2. Is mobile booking (1 = Mobile, 0 = Internet)
df_model['is_mobile'] = (df_model['sales_channel'] == 'Mobile').astype(int)

# 3. Is round trip
df_model['is_round_trip'] = (df_model['trip_type'] == 'RoundTrip').astype(int)

# 4. Early morning flight (before 6am — often business/frequent flyer)
df_model['is_early_flight'] = (df_model['flight_hour'] < 6).astype(int)

# 5. Weekend flight
df_model['is_weekend'] = df_model['flight_day'].isin(['Sat', 'Sun']).astype(int)

# 6. Short lead time (booked within 7 days — last minute)
df_model['is_last_minute'] = (df_model['purchase_lead'] <= 7).astype(int)

# 7. Long lead time (planned well ahead — over 90 days)
df_model['is_planned_ahead'] = (df_model['purchase_lead'] > 90).astype(int)

# 8. Encode flight_day as ordinal (Mon=0, Tue=1, ... Sun=6)
day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
df_model['flight_day_num'] = df_model['flight_day'].map(day_map)

# 9. Route frequency (high-frequency routes may have different booking patterns)
route_freq = df_model['route'].value_counts()
df_model['route_freq'] = df_model['route'].map(route_freq)

# 10. Booking origin frequency
origin_freq = df_model['booking_origin'].value_counts()
df_model['origin_freq'] = df_model['booking_origin'].map(origin_freq)

print("Feature engineering complete.")
print(f"Original features: {df.shape[1]-1}")
print(f"Total features after engineering: {df_model.shape[1]-1}")
print("\nNew features added:")
new_feats = ['total_addons', 'is_mobile', 'is_round_trip', 'is_early_flight',
             'is_weekend', 'is_last_minute', 'is_planned_ahead',
             'flight_day_num', 'route_freq', 'origin_freq']
for f in new_feats:
    print(f"  + {f}")


# ── CELL 6: Prepare Features for Modelling ───────────────────
# Drop columns not used in model
drop_cols = ['sales_channel', 'trip_type', 'flight_day', 'route',
             'booking_origin', 'booking_complete']

feature_cols = [c for c in df_model.columns if c not in drop_cols]
X = df_model[feature_cols]
y = df_model['booking_complete']

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeatures used:")
for f in feature_cols:
    print(f"  {f}")


# ── CELL 7: Train Random Forest with Cross-Validation ─────────
print("Training Random Forest with 5-fold Stratified Cross-Validation...")
print("This may take a minute...\n")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=10,
    class_weight='balanced',   # handles class imbalance (85/15 split)
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    rf, X, y, cv=cv,
    scoring=['accuracy', 'roc_auc', 'f1', 'precision', 'recall'],
    return_train_score=True
)

print("=== CROSS-VALIDATION RESULTS (5-fold) ===")
print(f"Accuracy:   {scores['test_accuracy'].mean():.4f}  ± {scores['test_accuracy'].std():.4f}")
print(f"ROC-AUC:    {scores['test_roc_auc'].mean():.4f}  ± {scores['test_roc_auc'].std():.4f}")
print(f"F1 Score:   {scores['test_f1'].mean():.4f}  ± {scores['test_f1'].std():.4f}")
print(f"Precision:  {scores['test_precision'].mean():.4f}  ± {scores['test_precision'].std():.4f}")
print(f"Recall:     {scores['test_recall'].mean():.4f}  ± {scores['test_recall'].std():.4f}")


# ── CELL 8: Fit Final Model on Full Data for Interpretation ───
rf.fit(X, y)
print("Final model fitted on full dataset.")


# ── CELL 9: Feature Importance Visualisation ─────────────────
importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)

# Clean display names
name_map = {
    'purchase_lead':       'Purchase Lead Time (days)',
    'flight_duration':     'Flight Duration (hrs)',
    'length_of_stay':      'Length of Stay (days)',
    'flight_hour':         'Flight Hour',
    'num_passengers':      'Number of Passengers',
    'route_freq':          'Route Frequency',
    'origin_freq':         'Booking Origin Frequency',
    'flight_day_num':      'Flight Day of Week',
    'total_addons':        'Total Add-ons Selected',
    'wants_extra_baggage': 'Wants Extra Baggage',
    'wants_in_flight_meals':'Wants In-flight Meals',
    'wants_preferred_seat':'Wants Preferred Seat',
    'is_planned_ahead':    'Planned Ahead (>90 days)',
    'is_last_minute':      'Last Minute Booking',
    'is_round_trip':       'Is Round Trip',
    'is_weekend':          'Weekend Flight',
    'is_mobile':           'Mobile Booking',
    'is_early_flight':     'Early Morning Flight',
}
importances.index = [name_map.get(i, i) for i in importances.index]

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#C8102E' if v > importances.median() else '#003087'
          for v in importances.values]
bars = ax.barh(importances.index, importances.values * 100,
               color=colors, edgecolor='white', height=0.7)

# Value labels
for bar, val in zip(bars, importances.values * 100):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Feature Importance (%)', fontsize=11)
ax.set_title('Random Forest — Feature Importance\nBritish Airways Customer Booking Prediction',
             fontsize=13, fontweight='bold', pad=15)
ax.axvline(importances.median() * 100, color='grey', linestyle='--',
           linewidth=1, alpha=0.5, label='Median importance')

red_patch  = mpatches.Patch(color='#C8102E', label='Above median importance')
blue_patch = mpatches.Patch(color='#003087', label='Below median importance')
ax.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('ba_feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()
print("Feature importance chart saved.")


# ── CELL 10: Cross-Validation Metrics Bar Chart ───────────────
metrics = {
    'Accuracy':  scores['test_accuracy'].mean(),
    'ROC-AUC':   scores['test_roc_auc'].mean(),
    'F1 Score':  scores['test_f1'].mean(),
    'Precision': scores['test_precision'].mean(),
    'Recall':    scores['test_recall'].mean(),
}
errors = {
    'Accuracy':  scores['test_accuracy'].std(),
    'ROC-AUC':   scores['test_roc_auc'].std(),
    'F1 Score':  scores['test_f1'].std(),
    'Precision': scores['test_precision'].std(),
    'Recall':    scores['test_recall'].std(),
}

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(metrics.keys(), metrics.values(),
              yerr=errors.values(), capsize=5,
              color=['#003087','#C8102E','#D4AF37','#003087','#003087'],
              edgecolor='white', width=0.55)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('5-Fold Cross-Validation Performance Metrics\nBritish Airways Booking Prediction Model',
             fontsize=12, fontweight='bold')
ax.axhline(0.5, color='grey', linestyle='--', alpha=0.4, label='Chance level (0.5)')
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('ba_cv_metrics.png', bbox_inches='tight', dpi=150)
plt.show()


# ── CELL 11: ROC Curve ────────────────────────────────────────
from sklearn.model_selection import cross_val_predict
y_prob = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')[:,1]
fpr, tpr, _ = roc_curve(y, y_prob)
auc = roc_auc_score(y, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#003087', linewidth=2.5,
        label=f'Random Forest (AUC = {auc:.3f})')
ax.plot([0,1],[0,1], color='grey', linestyle='--', linewidth=1, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color='#003087')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve — Booking Prediction Model', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig('ba_roc_curve.png', bbox_inches='tight', dpi=150)
plt.show()


# ── CELL 12: Key Findings Summary ────────────────────────────
top3 = importances.tail(3)
print("""
================================================================
TASK 2 — KEY FINDINGS SUMMARY
================================================================

MODEL PERFORMANCE (5-fold Cross-Validation):
""")
for k, v in metrics.items():
    print(f"  {k:<12}: {v:.4f}  (±{errors[k]:.4f})")

print(f"""
CLASS IMBALANCE NOTE:
  The dataset is imbalanced — only 14.9% of customers complete
  a booking. We used class_weight='balanced' in the Random Forest
  and report F1/Precision/Recall rather than relying on accuracy alone.
  ROC-AUC of {metrics['ROC-AUC']:.3f} is a more reliable metric here.

TOP 3 MOST PREDICTIVE FEATURES:""")
for feat, imp in top3.items():
    print(f"  {feat}: {imp*100:.1f}%")

print("""
INTERPRETATION:
  1. PURCHASE LEAD TIME is the strongest predictor — customers
     who book far in advance are significantly more likely to
     complete their booking. This likely reflects committed,
     planned travellers vs casual browsers.

  2. FLIGHT DURATION has high importance — longer flights
     (international) show different booking completion patterns
     vs short domestic routes.

  3. LENGTH OF STAY is the third driver — customers with
     specific, planned holidays (longer stays) show higher
     completion rates than open-ended or last-minute travellers.

COMMERCIAL IMPLICATION FOR BA:
  Target proactive marketing at customers who:
  • Have shown early planning intent (searched far in advance)
  • Are booking long-haul/international routes
  • Have selected add-ons (baggage, meals, seat) — strong signal
    of purchase intent
================================================================
""")
