import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. SETUP AND DATA INGESTION
# Adjust the filename here if your file name differs
file_name = 'Customer Purchase Behavior datasets.xlsx'

def run_supply_chain_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please check your GitHub sidebar.")
        return

    # Load Data
    df = pd.read_excel(file_path)
    
    # 2. PHASE 2: DEMAND INTELLIGENCE
    demand_summary = df.groupby('Product_Category')['Total_Spent'].sum().sort_values(ascending=False)
    top_cat = demand_summary.index[0]
    top_rev = demand_summary.iloc[0]

    # 3. PHASE 3: OPERATIONAL EFFICIENCY
    df['RPM'] = df['Total_Spent'] / df['Browsing_Time_Before_Purchase']
    peak_window = df.groupby('Preferred_Shopping_Time')['RPM'].median().idxmax()
    cold_pct = (len(df[df['Last_Purchase_Days_Ago'] > 90]) / len(df)) * 100

    # 4. PHASE 4: PREDICTIVE MODELING
    threshold = df['Total_Spent'].quantile(0.70)
    df['Is_High_Value'] = (df['Total_Spent'] > threshold).astype(int)
    
    le = LabelEncoder()
    df['Loc_Enc'] = le.fit_transform(df['Location'])
    df['Seg_Enc'] = le.fit_transform(df['Customer_Segment'])
    
    features = ['Age', 'Income', 'Loyalty_Points_Used', 'Previous_Purchases', 
                'Average_Spending', 'Browsing_Time_Before_Purchase', 'Loc_Enc', 'Seg_Enc']
    
    X = df[features]
    y = df['Is_High_Value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    f1 = f1_score(y_test, model.predict(X_test))
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    top_feature = importances.index[0]

    # 5. OUTPUT GENERATOR FOR README
    print("\n" + "="*60)
    print("COPY THE TEXT BELOW INTO YOUR README.MD")
    print("="*60 + "\n")
    print("## Strategic Inferences & Findings\n")
    print(f"### 1. Revenue Concentration (Pareto)")
    print(f"* Primary Driver: The '{top_cat}' category is the leading revenue generator, totaling ${top_rev:,.2f}.")
    print(f"* Strategy: Buffer inventory levels for this category by 15% to mitigate stockout risks.\n")
    print(f"### 2. Operational Throughput")
    print(f"* Peak Performance Window: Highest throughput (Revenue per Minute) occurs during '{peak_window}' shopping periods.")
    print(f"* Strategy: Prioritize fulfillment labor and logistics bandwidth during this window.\n")
    print(f"### 3. Inventory Risk Management")
    print(f"* Cold Demand Signal: {cold_pct:.1f}% of the customer base is 'Cold' (No purchase > 90 days).")
    print(f"* Strategy: Implement markdown strategies or re-allocation to 'Active' segments.\n")
    print(f"### 4. Predictive Decision Intelligence")
    print(f"* Core Value Indicator: '{top_feature}' is the strongest predictor of high-value transactions.")
    print(f"* Strategy: Deploy the F1-Score: {f1:.2f} prediction engine to pre-assign premium shipping lanes.\n")
    print("="*60)

    # Exporting for GitHub
    df.to_csv('supply_chain_action_plan.csv', index=False)

# Execute the engine
run_supply_chain_analysis(file_name)
