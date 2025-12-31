import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Ensure high-resolution rendering in the notebook
%matplotlib inline
sns.set_theme(style="whitegrid")

# Create Directory for GitHub README
if not os.path.exists('images'):
    os.makedirs('images')

# LOAD DATA
# Using the exact name from your previous file list
file_name = 'Customer Purchase Behavior datasets.xlsx'
df = pd.read_excel(file_name)

# --- PHASE 2: CATEGORY PARETO ---
plt.figure(figsize=(10, 6))
demand = df.groupby('Product_Category')['Total_Spent'].sum().sort_values(ascending=False)
sns.barplot(x=demand.index, y=demand.values, palette='viridis')
plt.title('Demand Intelligence: Revenue by Category')
plt.xticks(rotation=45)
plt.savefig('images/category_pareto.png', bbox_inches='tight')
plt.show() # Forces the image to appear in the notebook

# --- PHASE 4: FEATURE IMPORTANCE ---
# Prep data for modeling
le = LabelEncoder()
df['Location_Enc'] = le.fit_transform(df['Location'])
df['Segment_Enc'] = le.fit_transform(df['Customer_Segment'])
features = ['Age', 'Income', 'Loyalty_Points_Used', 'Previous_Purchases', 
            'Average_Spending', 'Browsing_Time_Before_Purchase', 'Customer_Satisfaction_Rating', 'Location_Enc', 'Segment_Enc']

X = df[features]
y = (df['Total_Spent'] > df['Total_Spent'].quantile(0.7)).astype(int)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances.values, y=importances.index, palette='rocket')
plt.title('Predictive Decision Drivers')
plt.savefig('images/feature_importance.png', bbox_inches='tight')
plt.show()

# --- PHASE 5: THE COMMAND CENTER ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4)

# 1. Revenue
sns.barplot(ax=axes[0,0], x=demand.index, y=demand.values, palette='viridis')
axes[0,0].set_title('Revenue per Category')

# 2. Efficiency
sns.scatterplot(ax=axes[0,1], data=df, x='Browsing_Time_Before_Purchase', y='Total_Spent', hue='Customer_Satisfaction_Rating', palette='RdYlGn')
axes[0,1].set_title('Throughput: Time vs Spend')

# 3. Predictors
sns.barplot(ax=axes[1,0], x=importances.head(5).values, y=importances.head(5).index, palette='magma')
axes[1,0].set_title('Top 5 Value Predictors')

# 4. Inventory Freshness
df['Last_Purchase_Days_Ago'].apply(lambda x: 'Active' if x <= 30 else 'Cold').value_counts().plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
axes[1,1].set_title('Demand Freshness')

plt.suptitle('Supply Chain Operational Dashboard', fontsize=20)
plt.savefig('images/operational_dashboard.png', bbox_inches='tight', dpi=300)
plt.show()

print("Verification: Check the 'images' folder in your sidebar. The files must exist there for the README to work.")
