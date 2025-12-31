# ==========================================
# 0. INFRASTRUCTURE & DATA INGESTION
# ==========================================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Structural Improvement: Define filename based on repository contents
file_name = 'Customer Purchase Behavior datasets.xlsx'

print("--- Initializing Supply Chain Engine ---")

try:
    # Logic: Read Excel file directly for high-fidelity data ingestion
    df = pd.read_excel(file_name)
    print(f"Success: {len(df)} records ingested.")
except Exception as e:
    print(f"Error: File '{file_name}' not found. Verify it is in your main directory.")

# ==========================================
# 1. PHASE 2: DEMAND INTELLIGENCE (Pareto)
# ==========================================
# Identifying the primary revenue drivers
demand_summary = df.groupby('Product_Category')['Total_Spent'].sum().sort_values(ascending=False)
top_category = demand_summary.index[0]
top_category_rev = demand_summary.iloc[0]

# ==========================================
# 2. PHASE 3: OPERATIONAL EFFICIENCY
# ==========================================
# Calculating Revenue per Minute (RPM) to measure throughput velocity
df['RPM'] = df['Total_Spent'] / df['Browsing_Time_Before_Purchase']
efficiency = df.groupby('Preferred_Shopping_Time')['RPM'].median().sort_values(ascending=False)
peak_window = efficiency.index[0]

# Recency Analysis: Identifying inventory risk
# Segmenting 'Cold' demand (No purchase > 90 days)
cold_count = len(df[df['Last_Purchase_Days_Ago'] > 90])
cold_pct = (cold_count / len(df)) * 100

# ==========================================
# 3. PHASE 4: PREDICTIVE MODELING
# ==========================================
# Logic: Define High-Value targets (Top 30% of spenders)
threshold = df['Total_Spent'].quantile(0.70)
df['Is_High_Value'] = (df['Total_Spent'] > threshold).astype(int)

# Structural Improvement: Categorical Encoding for Machine Learning
le = LabelEncoder()
df['Loc_Enc'] = le.fit_transform(df['Location'])
df['Seg_Enc'] = le.fit_transform(df['Customer_Segment'])

features = ['Age', 'Income', 'Loyalty_Points_Used', 'Previous_Purchases', 
            'Average_Spending', 'Browsing_Time_Before_Purchase', 'Loc_Enc', 'Seg_Enc']
X = df[features]
y = df['Is_High_Value']

# Train-Test Split for validation integrity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Metrics & Feature Importance
f1 = f1_score(y_test, model.predict(X_test))
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
primary_driver = importances.index[0]

# ==========================================
# 4. PHASE 5: STRATEGIC INFERENCE GENERATOR
# ==========================================
print("\n" + "="*60)
print("     STRATEGIC INFERENCES FOR YOUR README")
print("="*60 + "\n")

print(f"## Strategic Inferences & Findings\n")

print(f"### 1. Revenue Concentration (Pareto)")
print(f"* **Primary Driver:** The '{top_category}' category is the leading revenue generator, totaling ${top_category_rev:,.2f}.")
print(f"* **Strategy:** Buffer inventory levels for this category by 15% to mitigate stockout risks.\n")

print(f"### 2. Operational Throughput")
print(f"* **Peak Performance Window:** Highest throughput (Revenue per Minute) occurs during '{peak_window}' shopping periods.")
print(f"* **Strategy:** Prioritize fulfillment labor and logistics bandwidth during this window to maximize capital velocity.\n")

print(f"### 3. Inventory Risk Management")
print(f"* **Cold Demand Signal:** {cold_pct:.1f}% of the customer base is 'Cold' (No purchase > 90 days).")
print(f"* **Strategy:** This inventory represents capital stagnation; implement markdown strategies or re-allocation to 'Active' segments.\n")

print(f"### 4. Predictive Decision Intelligence")
print(f"* **Core Value Indicator:** '{primary_driver}' is the strongest predictor of high-value transactions.")
print(f"* **Strategy:** Deploy the F1-Score: {f1:.2f} prediction engine to pre-assign premium shipping lanes for customers matching this profile.\n")

print("="*60)

# Export high-alpha data
df['High_Value_Prob'] = model.predict_proba(X)[:, 1]
df.to_csv('supply_chain_action_plan.csv', index=False)
print("\n[Action] File 'supply_chain_action_plan.csv' has been exported for logistics routing.")
