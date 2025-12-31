# ==========================================
# STEP 0: INFRASTRUCTURE SETUP
# ==========================================
print("Initializing environment...")
!pip install pandas numpy matplotlib seaborn openpyxl scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ==========================================
# PHASE 1: DATA INTEGRITY (Source of Truth)
# ==========================================
file_name = 'Customer Purchase Behavior datasets.xlsx'

try:
    df = pd.read_excel(file_name)
    print("--- Phase 1: Data Loaded Successfully ---")
    print(f"Dataset Shape: {df.shape}")
    
    # Check for missing values
    if df.isnull().sum().sum() == 0:
        print("Data Integrity: Clean (No missing values)")
    else:
        print("Data Integrity: Missing values detected and handled.")
        df = df.fillna(df.median(numeric_only=True))
except Exception as e:
    print(f"CRITICAL ERROR: Could not find '{file_name}'. Ensure it is in the main folder.")

# ==========================================
# PHASE 2: DEMAND INTELLIGENCE (Pareto)
# ==========================================
demand_summary = df.groupby('Product_Category').agg({
    'Quantity': 'sum',
    'Total_Spent': 'sum',
    'Transaction_ID': 'count'
}).sort_values(by='Total_Spent', ascending=False)

# Create and Save Pareto Visual
plt.figure(figsize=(10, 6))
sns.barplot(x=demand_summary.index, y=demand_summary['Total_Spent'], palette='viridis')
plt.title('Phase 2: Revenue by Product Category')
plt.xticks(rotation=45)
plt.savefig('category_pareto.png', bbox_inches='tight', dpi=300)
print("--- Phase 2 Complete: category_pareto.png saved ---")

# ==========================================
# PHASE 3: EFFICIENCY & THROUGHPUT
# ==========================================
# Revenue per Minute calculation
df['Revenue_Per_Minute'] = df['Total_Spent'] / df['Browsing_Time_Before_Purchase']

# Recency Labeling (Supply Chain Freshness)
def recency_label(days):
    if days <= 30: return '1. Active'
    elif days <= 90: return '2. Warming'
    else: return '3. Cold'
df['Recency_Status'] = df['Last_Purchase_Days_Ago'].apply(recency_label)
recency_counts = df['Recency_Status'].value_counts().sort_index()

print("--- Phase 3 Complete: Efficiency metrics calculated ---")

# ==========================================
# PHASE 4: PREDICTIVE MODELING (Decision Intelligence)
# ==========================================
# Target: High-Value Transaction (Top 30%)
threshold = df['Total_Spent'].quantile(0.70)
df['Is_High_Value'] = (df['Total_Spent'] > threshold).astype(int)

# Feature Engineering
features = ['Age', 'Income', 'Loyalty_Points_Used', 'Previous_Purchases', 
            'Average_Spending', 'Browsing_Time_Before_Purchase', 'Customer_Satisfaction_Rating']
le = LabelEncoder()
df['Location_Encoded'] = le.fit_transform(df['Location'])
df['Segment_Encoded'] = le.fit_transform(df['Customer_Segment'])
X_features = features + ['Location_Encoded', 'Segment_Encoded']

X = df[X_features]
y = df['Is_High_Value']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Feature Importance Plot
importances = pd.DataFrame({'Feature': X_features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=importances, x='Importance', y='Feature', palette='rocket')
plt.title('Phase 4: Drivers of High-Value Transactions')
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
print("--- Phase 4 Complete: feature_importance.png saved ---")

# ==========================================
# PHASE 5: COMMAND CENTER DASHBOARD
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.4)

# A. Category Revenue
sns.barplot(ax=axes[0, 0], x=demand_summary.index, y=demand_summary['Total_Spent'], palette='viridis')
axes[0, 0].set_title('Revenue by Category')

# B. Efficiency Mapping
sns.scatterplot(ax=axes[0, 1], data=df, x='Browsing_Time_Before_Purchase', y='Total_Spent', hue='Customer_Satisfaction_Rating', palette='RdYlGn', alpha=0.5)
axes[0, 1].set_title('Efficiency: Browsing Time vs. Spend')

# C. Top 5 Predictive Drivers
sns.barplot(ax=axes[1, 0], data=importances.head(5), x='Importance', y='Feature', palette='magma')
axes[1, 0].set_title('Predictive Decision Drivers')

# D. Inventory Freshness (Recency)
recency_counts.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#f1c40f', '#e74c3c'])
axes[1, 1].set_title('Demand Freshness (Active to Cold)')

plt.suptitle('SUPPLY CHAIN OPERATIONAL DASHBOARD', fontsize=20, fontweight='bold')
plt.savefig('operational_dashboard.png', bbox_inches='tight', dpi=400)
plt.show()

# Final Alpha Export
df['High_Value_Probability'] = model.predict_proba(X)[:, 1]
df.to_csv('supply_chain_action_plan.csv', index=False)
print("\n--- ALL PHASES COMPLETE ---")
print("1. Dashboard: operational_dashboard.png saved")
print("2. Action Plan: supply_chain_action_plan.csv exported")
print(f"3. High-Value Prediction Accuracy: {model.score(X_test, y_test):.2%}")
