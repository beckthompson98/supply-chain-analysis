# Supply Chain Analysis & Predictive Demand Optimization

## Executive Summary
This project provides an end-to-end analysis of customer purchase behavior to optimize regional distribution strategies. By integrating machine learning and classical inventory management techniques, the system identifies high-value demand signals to streamline supply chain resource allocation.

## Project Structure
- **/data**: Houses the raw dataset and the generated `supply_chain_action_plan.csv`.
- **/notebooks**: Contains `analysis.ipynb`, covering the full data pipeline from preprocessing to predictive modeling.

## Technical Logic & Inferences
The analysis utilized a **Random Forest Classifier** to predict high-value customer segments. Based on the feature importance and model evaluation, the following inferences were identified:

* **Primary Demand Drivers**: Customer **Income** and **Average Spending** were the most significant predictors of high-value status.
* **Secondary Signals**: **Loyalty Points Used** and **Age** provided meaningful secondary signals for predicting purchase frequency.
* **Operational Outcome**: The model allows for the automated creation of a distribution action plan, prioritizing regions and segments that contribute most to the Pareto "A" category (the top 20% of contributors).
* **Model Accuracy**: The current model achieves an overall accuracy of **68%**.

## Tech Stack
* **Language**: Python 3.12
* **Libraries**: Pandas, Scikit-Learn (RandomForestClassifier)
* **Environment**: GitHub Codespaces
