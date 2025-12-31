# ==========================================
# 0. INFRASTRUCTURE & SETUP
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Force inline plotting for cloud environments
%matplotlib inline
plt.rcParams['figure.facecolor'] = 'white'

# Create images directory for local storage (optional)
if not os.path.exists('images'):
    os.makedirs('images')

# ==========================================
# 1. DATA INTEGRITY (Phase 1)
# ==========================================
file_name = 'Customer Purchase Behavior datasets.xlsx'

try:
    df = pd.read_excel(file_name)
    print(f"--- Phase 1 Complete ---")
    print(f"Source: {file_name} | Records: {len(df)}")
except Exception as e:
    print(f"Error: Could not load {file_name}. Check the file sidebar.")

# ==========================================
# 2. DEMAND INTELLIGENCE (Phase 2)
# ==========================================
demand = df.groupby('Product_Category')['Total_Spent'].sum().sort_values(ascending=False)
top_category = demand.index[0]
top_category_rev = demand.iloc[0]

# Visualize for the session
plt.figure(figsize=(10, 6))
sns.barplot(x=demand.index, y=demand.values, palette='viridis')
plt.title('Revenue by Product Category')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)
plt.savefig('images/category_pareto.png', bbox_inches='tight')
plt.show()

# ==========================================
# 3. EFFICIENCY & RECENTY (Phase 3)
# ==========================================
df['Revenue_Per_Minute'] = df['Total_Spent'] / df['Browsing_Time_Before_Purchase']
efficiency = df.groupby('Preferred_Shopping_Time')['Revenue_Per_Minute'].median().sort_values(ascending=False)
peak_time = efficiency.index[0]

def get_recency(days):
    if days <= 30: return 'Active'
    elif days <= 90: return 'Warming'
    else: return 'Cold'

df['Recency_Status'] = df['Last_Purchase_Days_Ago'].apply(get_recency)
cold_pct = (df['Recency_Status'] == 'Cold').mean() * 100

# ==========================================
# 4. PREDICTIVE MODELING (Phase 4)
# ==========================================
threshold = df['Total_Spent'].quantile(0.70)
df['Is_High_Value'] = (df['Total_Spent'] > threshold).astype(int)

le = LabelEncoder()
df['Loc_Enc'] = le.fit_transform(df['Location'])
df['Seg_Enc'] = le.fit_transform(df['Customer_Segment'])

features = ['Age', 'Income', 'Loyalty_Points_Used', 'Previous_Purchases', 
            'Average_Spending', 'Browsing_Time_Before_Purchase', 'Customer_Satisfaction_Rating',
            'Loc_Enc', 'Seg_Enc']

X = df[features]
y = df['Is_High_Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
top_feature = importances.index[0]
model_f1 = f1_score(y_test, model.predict(X_test))

# ==========================================
# 5. COMMAND CENTER DASHBOARD (Phase 5)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4)

sns.barplot(ax=axes[0,0], x=demand.index, y=demand.values, palette='viridis')
axes[0,0].set_title('Revenue per Category')

sns.scatterplot(ax=axes[0,1], data=df, x='Browsing_Time_Before_Purchase', y='Total_Spent', hue='Customer_Satisfaction_Rating', palette='RdYlGn')
axes[0,1].set_title('Throughput: Time vs Spend')

sns.barplot(ax=axes[1,0], x=importances.head(5).values, y=importances.head(5).index, palette='magma')
axes[1,0].set_title('Top 5 Value Predictors')

df['Recency_Status'].value_counts().plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%', colors=['#2ecc71', '#f1c40f', '#e74c3c'])
axes[1,1].set_title('Demand Freshness (Recency)')

plt.suptitle('Supply Chain Operational Dashboard', fontsize=20, fontweight='bold')
plt.savefig('images/operational_dashboard.png', bbox_inches='tight')
plt.show()

# Final CSV Export
df['High_Value_Prob'] = model.predict_proba(X)[:, 1]
df.to_csv('supply_chain_action_plan.csv', index=False)

# ==========================================
# STRATEGIC INFERENCES (The "New Part")
# ==========================================
print("\n" + "="*60)
print("             STRATEGIC INFERENCES FOR README")
print("="*60)
print(f"1. Revenue Concentration: '{top_category}' is the primary driver, generating ${top_category_rev:,.2f}.")
print(f"2. Decisive Buyer Window: Customers shopping during '{peak_time}' show the highest RPM.")
print(f"3. Inventory Risk: {cold_pct:.1f}% of the customer base is 'Cold' (High risk of obsolescence).")
print(f"4. Predictive Power: '{top_feature}' is the strongest indicator of customer value.")
print(f"5. Model Reliability: The prediction engine achieved an F1-Score of {model_f1:.2f}.")
print("="*60)
