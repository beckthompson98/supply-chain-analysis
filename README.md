# High-Performance Supply Chain Analysis: Demand & Decision Intelligence

## Executive Summary
This project applies a logic-driven mental model framework to optimize supply chain operations using a dataset of customer purchase behavior. By moving through five distinct phases—from Data Integrity to Predictive Modeling—this analysis reduces Information Entropy and provides actionable Alpha for logistics and inventory management.

## Mental Models Applied
* **The Pareto Principle (80/20):** Identifying the 20% of product categories driving 80% of revenue to prioritize stock security.
* **Entropy Reduction:** Cleaning data signals to ensure the "Source of Truth" is 100% reliable for downstream automation.
* **Throughput Efficiency:** Measuring the velocity of transactions relative to browsing time to identify checkout friction.

## Tech Stack
* **Environment:** GitHub Codespaces / Jupyter
* **Engine:** Python 3.x
* **Libraries:** Pandas (Data Manipulation), Scikit-Learn (Predictive Modeling), Seaborn/Matplotlib (Strategic Visualization), Openpyxl (Excel Ingestion).

## The 5-Phase Framework

### Phase 1: Data Integrity (Source of Truth)
Established a clean pipeline by auditing missing values and duplicate records. 
* **Result:** Analyzed [INSERT TOTAL RECORDS] transactions with 0% data loss.

### Phase 2: Demand Intelligence (Pareto Analysis)
Segmented revenue by category and location to identify "Cash Cows" vs. "Workhorses."
* **Key Insight:** [INSERT TOP CATEGORY] is the primary revenue driver.

![Category Performance Matrix](images/category_pareto.png)

### Phase 3: Operational Efficiency (Throughput)
Calculated the Revenue per Minute ($R_{pm}$) metric to find peak operational windows.
* **Key Insight:** [INSERT TOP TIME] shoppers have the highest $R_{pm}$, suggesting a high-velocity fulfillment requirement.

### Phase 4: Predictive Modeling (Decision Intelligence)
Built a Random Forest Classifier to predict "High-Value Transactions" before they finalize.
* **Performance:** Achieved an F1-Score of [INSERT F1-SCORE].
* **Top Predictor:** [INSERT TOP FEATURE] was identified as the strongest leading indicator of customer value.

![Feature Importance Drivers](images/feature_importance.png)

### Phase 5: The Command Center
Synthesized all signals into a 4-quadrant dashboard for executive decision-making.

![Supply Chain Operational Dashboard](images/operational_dashboard.png)

## Strategic Results
* **Inventory Freshness:** Identified that [INSERT COLD %]% of the customer base is "Cold," signaling a need for inventory re-allocation.
* **Logistics Priority:** The system now assigns a `High_Value_Probability` score to every customer, allowing for Predictive Routing of premium shipments.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn openpyxl`.
3. Open `analysis.ipynb` and run all cells.
