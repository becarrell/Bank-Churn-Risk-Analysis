"""
================================================================================
BANK CUSTOMER CHURN PREDICTION & ANALYSIS
================================================================================

Overview:
- Bank churn data from Kaggle, cleaned in PostgreSQL, and analyzed in Python
- Data cleaned and feature engineered in SQL
- Python used for EDA, visualization, modeling, and business insights
- Compares rule-based churn risk (from SQL) vs. machine learning approach
================================================================================
"""

# ================================================================================
# IMPORTS & CONFIGURATION
# ================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Visual settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("BANK CUSTOMER CHURN ANALYSIS")
print("="*80)

# ================================================================================
# DATA LOADING & VALIDATION
# ================================================================================

# Load cleaned dataset from SQL export
df = pd.read_csv(r"filepath here")

# Standardize column names (lowercase, underscores)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Data quality checks
missing = df.isnull().sum().sum()
duplicates = df.duplicated().sum()

print(f"Missing values: {missing}")
print(f"Duplicate rows: {duplicates}")

# Target variable overview
churn_rate = df['has_churned'].mean() * 100
print(f"Overall churn rate: {churn_rate:.1f}%")

# ================================================================================
# EXPLORATORY DATA ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Summary statistics by churn status
print("\n--- Key Metrics by Churn Status ---\n")
summary_cols = ['age', 'credit_score', 'balance', 'salary', 'tenure',
                'number_of_products', 'satisfaction_score', 'clv_estimated']

summary = df.groupby('has_churned')[summary_cols].mean()
summary.index = ['Not Churned', 'Churned']
print(summary.round(2))

# Key insight: What differs between churners and non-churners?
print("\n--- Percentage Differences (Churned vs Not Churned) ---\n")
pct_diff = ((summary.loc['Churned'] - summary.loc['Not Churned']) /
            summary.loc['Not Churned'] * 100)
print(pct_diff.sort_values(ascending=False).round(1))

# ================================================================================
# CORRELATION ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("Correlation Analysis")
print("=" * 80)

# Identify useful columns, has_complained is not used due to near one-to-one correlation with churn
print("\n--- Correlations with Churn (Strongest to Weakest) ---\n")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_vars = [col for col in numeric_cols if col not in ['customer_id', 'has_complained']]

correlations = df[corr_vars].corr()['has_churned'].drop('has_churned').sort_values(ascending=False)
print(correlations)

corr_matrix = df[corr_vars].corr()

print("\n--- Correlations with CLV (Ranked) ---")
clv_corr = corr_matrix['clv_estimated'].drop('clv_estimated').sort_values(ascending=False)
print(clv_corr)

# Visualization: Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    df[corr_vars].corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 01_correlation_heatmap.png")

# ================================================================================
# VIOLIN PLOTS
# ================================================================================

key_features = ['age', 'balance', 'number_of_products', 'satisfaction_score']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    sns.violinplot(
        data=df,
        x='has_churned',
        y=feature,
        ax=axes[idx],
        palette=['#186c3c', '#890000'],
        inner='quartile'
    )
    axes[idx].set_xlabel('Churned', fontweight='bold')
    axes[idx].set_ylabel(feature.replace('_', ' ').title(), fontweight='bold')
    axes[idx].set_xticklabels(['No', 'Yes'])
    axes[idx].set_title(f'{feature.replace("_", " ").title()} Distribution',
                        fontweight='bold')

fig.suptitle('Key Feature Distributions by Churn Status',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 02_feature_distributions.png")

# ================================================================================
# MARGINAL HISTOGRAM (AGE VS BALANCE BY RISK)
# (top correlations with churn)
# ================================================================================

g = sns.jointplot(
    data=df,
    x='age',
    y='balance',
    hue='churn_risk',
    palette={'Low Risk': '#186c3c', 'Medium Risk': '#f39c12',
             'High Risk': '#b8151a', 'Churned': '#890000'},
    alpha=0.6
)
g.figure.suptitle("Age vs Balance by Churn Risk Category", y=0.95, fontweight='bold')
plt.savefig('03_age_balance_risk.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 03_age_balance_risk.png")

# ================================================================================
# RULE-BASED CHURN RISK EVALUATION (FROM SQL)
# ================================================================================

print("\n" + "=" * 80)
print("RULE-BASED CHURN RISK MODEL (SQL LOGIC)")
print("=" * 80)

# The churn_risk column was calculated in SQL using engineered rules

print("\n--- Rule-Based Model Performance ---\n")

# Analyze churn rate by risk category
churn_by_risk = df.groupby('churn_risk').agg({
    'has_churned': ['count', 'sum', 'mean']
}).round(3)
churn_by_risk.columns = ['Total_Customers', 'Churned', 'Churn_Rate']
churn_by_risk['Churn_Rate'] = churn_by_risk['Churn_Rate'] * 100

# Sort by churn rate (validates if risk categories make sense)
churn_by_risk = churn_by_risk.sort_values('Churn_Rate', ascending=False)
print(churn_by_risk)

# For active customers only, convert risk to binary prediction
active_customers = df[df['churn_risk'] != 'Churned'].copy()
active_customers['risk_prediction'] = (
    active_customers['churn_risk'] == 'High Risk'
).astype(int)
# warning due to unconfirmed datatype from PyCharm

# Evaluate rule-based predictions
y_true_rules = active_customers['has_churned']
y_pred_rules = active_customers['risk_prediction']

print("\n--- Classification Performance (High Risk = Predicted Churn) ---\n")
print(classification_report(y_true_rules, y_pred_rules,
                           target_names=['Not Churned', 'Churned']))

rules_accuracy = (y_true_rules == y_pred_rules).mean()
print(f"Overall Accuracy: {rules_accuracy:.3f}")

# ===========================================
# CUSTOMER SEGMENTATION WITH DBSCAN
# ===========================================

cluster_features = [
    'age',
    'tenure',
    'credit_score',
    'balance',
    'salary',
    'number_of_products',
    'is_active',
    'has_card',
    'has_complained',
    'satisfaction_score',
    'points_earned',
    'clv_estimated',
    'is_female'
]

X_cluster = df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

dbscan = DBSCAN(
    eps=2.1,
    min_samples=30
)

df['cluster'] = dbscan.fit_predict(X_scaled)

print("\n--- CLUSTER DISTRIBUTION ---\n")
print(df['cluster'].value_counts())

cluster_profile = (
    df.groupby('cluster')[cluster_features + ['has_churned']]
      .mean()
      .round(2)
)

cluster_profile['churn_rate'] = (
    df.groupby('cluster')['has_churned'].mean().round(2)
)

print("\n--- CLUSTER PROFILES ---\n")
print(cluster_profile)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    learning_rate='auto'
)

tsne_results = tsne.fit_transform(X_scaled)

df['tsne_1'] = tsne_results[:, 0]
df['tsne_2'] = tsne_results[:, 1]

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x='tsne_1',
    y='tsne_2',
    hue='cluster',
    style='has_churned',
    palette='tab10',
    alpha=0.7
)

plt.title(
    'Customer Segmentation via DBSCAN (t-SNE Projection)',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Cluster / Churn')

plt.tight_layout()

plt.savefig('04_customer_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 04_customer_clusters.png")

# ================================================================================
# LOGISTIC REGRESSION
# ================================================================================

print("\n" + "="*80)
print("LOGISTIC REGRESSION")
print("="*80)

features = [col for col in numeric_cols if col not in ['customer_id', 'has_complained', 'has_churned']]

X = df[features]
y = df['has_churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

# Convert to DataFrame to preserve feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

# Add constant with proper naming
X_train_sm = sm.add_constant(X_train_scaled_df)
logit_model = sm.Logit(y_train, X_train_sm)
logit_results = logit_model.fit(disp=0)

print(logit_results.summary())

# VIF for multicollinearity check
vif_df = pd.DataFrame({
    'Feature': features,
    'VIF': [variance_inflation_factor(X_train_scaled, i) for i in range(len(features))]
}).sort_values('VIF', ascending=False)

print("\n--- VIF Scores ---")
print(vif_df)

X_test_sm = sm.add_constant(X_test_scaled_df)
y_pred_prob_lr = logit_results.predict(X_test_sm)
y_pred_lr = (y_pred_prob_lr >= 0.5).astype(int)

lr_auc = roc_auc_score(y_test, y_pred_prob_lr)
print(f"\nLogistic Regression ROC-AUC: {lr_auc:.3f}")

print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, y_pred_lr, target_names=['Not Churned', 'Churned']))

# Coefficient plot
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': logit_results.params[1:]  # Exclude constant
}).sort_values('Coefficient')

plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.axvline(0, linestyle='--', color='black', linewidth=1)
plt.title('Logistic Regression Coefficients', fontweight='bold')
plt.xlabel('Log-Odds Impact on Churn')
plt.tight_layout()
plt.savefig('05_logit_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

# ================================================================================
# MACHINE LEARNING MODEL (RANDOM FOREST)
# ================================================================================

print("\n" + "=" * 80)
print("RANDOM FOREST CLASSIFIER")
print("=" * 80)

X = df[features]
y = df['has_churned']

print(f"\nFeatures used: {len(features)}")
print(f"Training samples: {len(X):,}")

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"\nTrain set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Train Random Forest
# Using balanced class weights to handle class imbalance
rf_model = RandomForestClassifier(
    n_estimators=200,        # Number of trees
    max_depth=10,            # Limit depth to avoid overfitting
    min_samples_leaf=20,     # Minimum samples per leaf
    random_state=42,
    class_weight='balanced', # Handle class imbalance
    n_jobs=-1                # Use all CPU cores
)

print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)
print("Model trained successfully")

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# ================================================================================
# MODEL EVALUATION & COMPARISON
# ================================================================================

print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

# Random Forest performance
print("\n--- Random Forest Classification Report ---\n")
print(classification_report(y_test, y_pred,
                           target_names=['Not Churned', 'Churned']))

rf_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {rf_roc_auc:.3f}")
print(f"  → Interpretation: Model correctly ranks churners {rf_roc_auc*100:.1f}% of the time")

# Cross-validation for robustness check
print("\n--- Cross-Validation (5-Fold) ---")
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')
print(f"Mean ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  → Model performance is consistent across folds")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Top 10 Most Important Features ---\n")
print(feature_importance.head(10).to_string(index=False))

# ================================================================================
# MODEL COMPARISON
# ================================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

# SQL Rules-based performance
churn_by_risk = df.groupby('churn_risk')['has_churned'].mean() * 100

active = df[df['churn_risk'] != 'Churned'].copy()
active['risk_pred'] = (active['churn_risk'] == 'High Risk').astype(int)
rules_auc = roc_auc_score(active['has_churned'], active['risk_pred'])

# Get Random Forest predictions on full dataset
df['rf_churn_prob'] = rf_model.predict_proba(df[features])[:, 1]

comparison = pd.DataFrame({
    'Model': ['SQL Rules', 'Random Forest'],
    'ROC-AUC': [f"{rules_auc:.3f}", f"{rf_roc_auc:.3f}"],
    'Type': ['Business Rules', 'Ensemble ML']
})

print("\n--- Model Performance Comparison ---")
print(comparison.to_string(index=False))

# Churn rate by deciles for each model
print("\n--- Model Discrimination: Churn Rate by Risk Decile ---")

# Random Forest deciles
df['rf_decile'] = pd.qcut(df['rf_churn_prob'], q=10, labels=False, duplicates='drop')
rf_decile_perf = df.groupby('rf_decile')['has_churned'].mean() * 100


# Combined decile performance
decile_comparison = pd.DataFrame({
    'Decile': range(10),
    'RF_Churn_Rate_%': rf_decile_perf.values
})

print(decile_comparison.round(1))

# Visualization: Churn rate lift comparison
plt.figure(figsize=(12, 6))
overall_churn = df['has_churned'].mean() * 100

plt.plot(rf_decile_perf.index, rf_decile_perf.values,
         marker='o', label='Random Forest', linewidth=2, color='#3498db')
plt.axhline(overall_churn, color='gray', linestyle='--',
            label=f'Overall Churn ({overall_churn:.1f}%)', linewidth=1.5)

plt.title('Model Discrimination: Churn Rate by Probability Decile',
          fontsize=14, fontweight='bold')
plt.xlabel('Decile (0 = Lowest Risk, 9 = Highest Risk)', fontweight='bold')
plt.ylabel('Churn Rate (%)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('06_model_discrimination.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: 06_model_discrimination.png")

# ================================================================================
# BUSINESS IMPACT
# ================================================================================

print("\n" + "="*80)
print("BUSINESS IMPACT")
print("="*80)

# Define retention rate
retention_rate = 0.15

# 1. SQL Rules approach
rules_high_risk = df[(df['churn_risk'] == 'High Risk') & (df['has_churned'] == 0)]
rules_clv = rules_high_risk['clv_estimated'].sum() * retention_rate
rules_count = len(rules_high_risk)

# 2. Random Forest approach
df['rf_predicted_churn'] = (df['rf_churn_prob'] >= 0.5).astype(int)
rf_churners = df[(df['rf_predicted_churn'] == 1) & (df['has_churned'] == 0)]
rf_clv = rf_churners['clv_estimated'].sum() * retention_rate
rf_count = len(rf_churners)

print(f"\n--- CLV Retention Scenario (15% Success Rate) ---\n")

impact_summary = pd.DataFrame({
    'Model': ['SQL Rules', 'Random Forest'],
    'Customers_Targeted': [rules_count, rf_count],
    'Total_CLV_At_Risk': [
        rules_high_risk['clv_estimated'].sum(),
        rf_churners['clv_estimated'].sum()
    ],
    'Potential_CLV_Saved': [rules_clv, rf_clv]
})

# Format for better display
impact_summary['Total_CLV_At_Risk'] = impact_summary['Total_CLV_At_Risk'].apply(lambda x: f"${x:,.0f}")
impact_summary['Potential_CLV_Saved'] = impact_summary['Potential_CLV_Saved'].apply(lambda x: f"${x:,.0f}")

print(impact_summary.to_string(index=False))

print(f"\n--- Random Forest vs SQL Rules ---")
print(f"Additional CLV saved: ${rf_clv - rules_clv:+,.0f}")
print(f"Improvement: {(rf_clv/rules_clv - 1)*100:+.1f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - visualizations saved")
print("=" * 80)

# Export
output_df = df[[
    'customer_id', 'age', 'tenure', 'credit_score', 'balance',
    'salary', 'number_of_products', 'has_card', 'is_active',
    'has_complained', 'satisfaction_score', 'points_earned',
    'clv_estimated', 'is_female', 'geography', 'cardtype',
    'age_bucket', 'credit_bucket', 'balance_bucket', 'clv_bucket',
    'churn_risk', 'has_churned', 'rf_churn_prob', 'rf_predicted_churn'
]]

output_df.to_csv('powerbi_churn_data.csv', index=False)

