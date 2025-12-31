# High-Performance Supply Chain Analysis: Demand & Decision Intelligence

## Project Overview
This repository contains a logic-driven supply chain analysis focused on identifying revenue drivers, operational friction, and predictive customer value. The project follows a structured five-phase framework designed to reduce information entropy and increase decision speed for logistics and inventory management.

## Mental Models Applied
* **The Pareto Principle (80/20):** Identifying the 20% of product categories driving 80% of revenue.
* **Structural Improvement:** Transitioning from static historical data to active predictive intelligence.
* **Throughput Efficiency:** Optimizing the relationship between customer engagement time and capital expenditure.

## Technical Framework
* **Environment:** Python 3.x, GitHub Codespaces
* **Key Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Data Source:** Customer Purchase Behavior Dataset (Excel)

## Analysis Phases

### Phase 1: Data Integrity
Established a clean source of truth by auditing transaction records and ensuring categorical consistency. 

### Phase 2: Demand Intelligence
Segmented total revenue by product category to identify core market drivers.

![Category Performance](images/category_pareto.png)

### Phase 3: Operational Throughput
Calculated Revenue per Minute (RPM) and demand recency to identify inventory risk.

### Phase 4: Predictive Decision Intelligence
Deployed a Random Forest Classifier to identify high-value transactions before finalization. This allows for predictive logistics routing and prioritized fulfillment.

![Predictive Drivers](images/feature_importance.png)

### Phase 5: Supply Chain Command Center
Synthesized all operational signals into a high-density executive dashboard.

![Operational Dashboard](images/operational_dashboard.png)

## Strategic Outcomes
* **Predictive Asset Allocation:** The system now assigns a probability score to every customer, enabling the prioritization of high-value shipments.
* **Risk Mitigation:** Real-time recency tracking identifies "Cold" demand segments for immediate inventory re-allocation.

## How to Execute
1. Clone the repository to a local or cloud environment.
2. Run `pip install -r requirements.txt` (or manually install pandas, scikit-learn, and seaborn).
3. Execute the `analysis.ipynb` notebook to generate updated reports and images.
