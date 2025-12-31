# High-Performance Supply Chain Analysis: Demand & Decision Intelligence

## Project Overview
This repository hosts an end-to-end supply chain analytics engine developed in Python. The project is built on a high-performance framework focusing on three core pillars: logic-based mental models, structural data improvements, and decision speed. By processing customer purchase behavior, the system identifies revenue drivers, detects operational friction, and predicts high-value transaction risk.

## Mental Models & Frameworks
* **The Pareto Principle (80/20):** Identifying the 20% of product categories and customer segments driving 80% of total revenue.
* **Information Entropy Reduction:** Standardizing raw datasets into a "Single Source of Truth" to eliminate noise in decision-making.
* **Throughput Efficiency:** Analyzing the velocity of capital by measuring Revenue per Minute (RPM) across different shopping windows.

## Technical Stack
* **Language:** Python 3.x
* **Environment:** GitHub Codespaces / Jupyter
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn, Openpyxl
* **Modeling:** Random Forest Classifier for predictive value scoring.

## Operational Phases

### Phase 1: Data Integrity
The ingestion pipeline audits raw Excel data, handles missing values, and ensures structural consistency. This ensures that all downstream logistics decisions are based on verified data.

### Phase 2: Demand Intelligence
A multi-dimensional analysis of revenue by product category and location. This phase identifies "Powerhouse" categories that require high-priority inventory replenishment.

### Phase 3: Operational Efficiency
Integration of time-based metrics and recency status. We distinguish between "Active," "Warming," and "Cold" demand to mitigate the risk of inventory obsolescence.

### Phase 4: Predictive Decision Intelligence
The system utilizes a Random Forest model to calculate the probability of a transaction being "High-Value." 
* **Outcome:** Provides a `High_Value_Probability` score for every record, allowing logistics teams to prioritize premium fulfillment lanes before orders are processed.

### Phase 5: Supply Chain Command Center
A unified reporting module that synthesizes all metrics into a four-quadrant dashboard. 

## Strategic Outcomes
* **Predictive Routing:** Logistics managers can use the exported `supply_chain_action_plan.csv` to route shipments based on predicted customer value.
* **Inventory Freshness:** Immediate identification of "Cold" demand segments (transactions over 90 days) for targeted liquidation or marketing intervention.

## Execution Instructions
To generate the operational dashboard and the predictive action plan:
1. Open `analysis.ipynb` in GitHub Codespaces or a Jupyter environment.
2. Ensure the dataset `Customer Purchase Behavior datasets.xlsx` is in the root directory.
3. Run all cells to initialize the engine.
4. **Visualizations:** The script will dynamically generate and display the Category Pareto, Feature Importance, and the Operational Command Center Dashboard.
5. **Exports:** An actionable CSV will be generated as `supply_chain_action_plan.csv`.
