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

## Project Structure and Workflow
This project follows a clear data pipeline aligned with the Zindi competition setup.

### Data Pipeline
1. **Raw Data**
   - Client information CSV  
   - Invoice and consumption CSV  

2. **Data Merging & Preparation**
   - Client-level aggregation  
   - Handling missing values and outliers  
   - Feature construction based on billing behavior  

3. **Exploratory Analysis & Modeling**
   - Fraud pattern exploration  
   - Model training and evaluation using ROC-AUC  

### Notebooks
- **fraud__data_merge.ipynb**  
  This notebook contains the **data ingestion and merging steps**.  
  It loads the two raw CSV files (client data and invoice/consumption data), performs cleaning, and merges them into a **single client-level dataset** suitable for analysis and modeling.  
  Initial feature aggregation and sanity checks are also performed in this notebook.

- **EDA-and-modeling_fraud.ipynb**  
  This notebook focuses on **exploratory data analysis (EDA)**, **feature engineering**, and **model development**.  
  It includes model training, cross-validation, and performance evaluation using **ROC-AUC**, which is the official competition metric.

### Modeling Workflow
- Client-level feature aggregation  
- Train–test split with class stratification  
- Handling class imbalance  
- Model evaluation using ROC-AUC  

This structured workflow ensures reproducibility and alignment with competition requirements.

## Exploratory Data Analysis (EDA)
Key findings from exploratory analysis include:
- Strong **class imbalance**, with fraudulent cases being rare  
- Highly skewed consumption distributions with extreme outliers  
- Significant overlap between fraud and non-fraud consumption patterns  
- Variables such as **number of invoices** and **billing time span** show higher association with fraud  

These insights motivate the use of robust evaluation metrics and careful feature engineering.

## Modeling Approach
- Client-level feature aggregation  
- Stratified train–test split to preserve class distribution  
- Feature scaling and preprocessing  
- Advanced imbalance handling techniques (SMOTE, Random Over-Sampling, NearMiss)  
- Logistic Regression with `class_weight="balanced"`  
- Cross-validation  
- Model evaluation using **ROC-AUC**  

The approach prioritizes interpretability, robustness to imbalance, and competition alignment.

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
- imbalanced-learn (`imblearn`)  

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
