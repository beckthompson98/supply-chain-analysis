# ==========================================
# 0. DIAGNOSTIC & SETUP
# ==========================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Force images to show in the notebook
%matplotlib inline
plt.rcParams['figure.facecolor'] = 'white'

print("--- System Check ---")
if not os.path.exists('images'):
    os.makedirs('images')
    print("Confirmed: 'images' folder created.")

# Verify data source
file_name = 'Customer Purchase Behavior datasets.xlsx'
if os.path.exists(file_name):
    print(f"Confirmed: Found {file_name}")
else:
    print(f"Error: {file_name} not found. Check your file sidebar.")

# ==========================================
# 1. LOAD & CLEAN
# ==========================================
df = pd.read_excel(file_name)
print(f"Data Source Loaded: {df.shape[0]} rows identified.")

# ==========================================
# 2. CATEGORY PARETO (PHASE 2)
# ==========================================
plt.figure(figsize=(10, 6))
demand = df.groupby('Product_Category')['Total_Spent'].sum().sort_values(ascending=False)
sns.barplot(x=demand.index, y=demand.values, palette='viridis')
plt.title('Demand Intelligence: Revenue by Category', fontsize=14)
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)

# Save BEFORE showing
plt.savefig('images/category_pareto.png', bbox_inches='tight', dpi=300)
plt.show() 
print("Step 2: Pareto chart generated and saved to images/category_pareto.png")

# ==========================================
# 3. PREDICTIVE DRIVERS (PHASE 4)
# ==========================================
# Rapid Encoding
le = LabelEncoder()
df['Loc_Enc'] = le.fit_transform(df['Location'])
df['Seg_Enc'] = le.fit_transform(df['Customer_Segment'])
features = ['Age', 'Income', 'Loyalty_Points_Used', 'Previous_Purchases', 'Average_Spending', 'Browsing_Time_Before_Purchase', 'Loc_Enc', 'Seg_Enc']

X = df[features]
y = (df['Total_Spent'] > df['Total_Spent'].median()).astype(int)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances.values, y=importances.index, palette='rocket')
plt.title('Predictive Drivers of Customer Value', fontsize=14)
plt.xlabel('Importance Score')

plt.savefig('images/feature_importance.png', bbox_inches='tight', dpi=300)
plt.show()
print("Step 3: Feature importance chart generated and saved to images/feature_importance.png")

# ==========================================
# 4. OPERATIONAL DASHBOARD (PHASE 5)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4)

# A. Revenue per Category
sns.barplot(ax=axes[0, 0], x=demand.index, y=demand.values, palette='viridis')
axes[0, 0].set_title('Revenue by Category', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=30)

# B. Efficiency Mapping
sns.scatterplot(ax=axes[0, 1], data=df, x='Browsing_Time_Before_Purchase', y='Total_Spent', hue='Customer_Satisfaction_Rating', palette='RdYlGn', alpha=0.6)
axes[0, 1].set_title('Throughput: Time vs Spend', fontweight='bold')

# C. Value Predictors
sns.barplot(ax=axes[1, 0], x=importances.head(5).values, y=importances.head(5).index, palette='magma')
axes[1, 0].set_title('Top 5 Value Predictors', fontweight='bold')

# D. Demand Freshness
df['Last_Purchase_Days_Ago'].apply(lambda x: 'Active' if x <= 30 else 'Cold').value_counts().plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1, 1].set_title('Inventory Demand Signal', fontweight='bold')

plt.suptitle('Supply Chain Operational Command Center', fontsize=20, fontweight='bold')
plt.savefig('images/operational_dashboard.png', bbox_inches='tight', dpi=300)
plt.show()
print("Step 4: Final Dashboard generated and saved to images/operational_dashboard.png")

# Verification check
print("\n--- FINAL VERIFICATION ---")
image_files = os.listdir('images')
print(f"Files currently in images folder: {image_files}")
