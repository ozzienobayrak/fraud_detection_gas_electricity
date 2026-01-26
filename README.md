# Fraud Detection in Electricity and Gas Consumption (Tunisia)

## Project Overview
This project addresses the problem of **fraud detection in electricity and gas consumption** using customer-level data from Tunisia. The work is based on a public machine learning challenge hosted on **Zindi**, where the objective is to predict fraudulent customers using historical billing and consumption behavior.

The solution focuses on **client-level feature aggregation**, handling **severe class imbalance**, and optimizing model performance using **ROC-AUC** as the main evaluation metric.

## Data Description
The dataset contains electricity and gas billing records with the following key entities:

### Client-Level Information
- `client_id`: Unique customer identifier  
- `district`, `region`: Geographic information  
- `client_catg`: Customer category  

### Invoice & Consumption Information
- Number of invoices per client  
- Consumption statistics (mean, max, standard deviation)  
- Billing period duration and frequency  

### Target Variable
- `target`: Binary fraud indicator  
  - `1` → Fraud  
  - `0` → Non-fraud  

All features are aggregated **by `client_id`**, which is the prediction unit.

## Exploratory Data Analysis (EDA)
Key findings from exploratory analysis:
- Strong **class imbalance**, with fraudulent cases being rare  
- Highly skewed consumption distributions and extreme outliers  
- Significant overlap between fraud and non-fraud consumption values  
- Variables such as **number of invoices** and **billing time span** show higher association with fraud  

These findings motivate the use of robust evaluation metrics and careful feature engineering.

## Modeling Approach
- Client-level feature aggregation  
- Stratified train–test split to preserve class distribution  
- Feature scaling and preprocessing 
- Advanced imbalance handling (SMOTE, ROS, NearMiss...)
- Logistic Regression with `class_weight="balanced"`  
- Cross-validation
- Model evaluation using **ROC-AUC**  

The approach prioritizes interpretability and robustness to imbalance.

## Evaluation Metric
**Primary Metric: ROC-AUC**

ROC-AUC is used because it:
- Is robust to class imbalance  
- Measures ranking quality rather than fixed classification thresholds  
- Matches the official competition evaluation metric  

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- imblearn


## Results
The results show that:
- Client-level aggregation significantly improves predictive signal  
- Simple linear models achieve competitive ROC-AUC scores  
- Billing frequency and exposure duration are strong fraud indicators  

## Future Work
- Time-series–based feature engineering  
- Tree-based models (XGBoost, LightGBM)  
  

## Source
**Fraud Detection in Electricity and Gas Consumption Challenge**  
Platform: Zindi  
https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge
