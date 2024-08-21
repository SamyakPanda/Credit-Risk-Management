# Credit Risk Management

## Overview
Develop a machine learning model to predict the probability of loan repayment using a comprehensive dataset from Home Credit.

## Data Source
The data is provided by Home Credit and is available on [Kaggle](https://www.kaggle.com/c/home-credit-default-risk).

## Data Description
- **application_train**: Contains loan application data with the `TARGET` column indicating repayment status (`0`: repaid, `1`: difficulties in repayment).
- **application_test**: Similar structure to `application_train` but without the `TARGET` column.
- **Bureau**: Information on clients' previous credits from other financial institutions.
- **Bureau_balance**: Monthly balance information for previous credits from the Credit Bureau.
- **previous_application**: Data on all previous loan applications at Home Credit.
- **POS_CASH_BALANCE**: Monthly balance of previous point of sale (POS) loans.
- **credit_card_balance**: Monthly balance snapshots of previous credit cards.
- **installments_payments**: Repayment history for previously disbursed credits.

## Preprocessing
- **Aggregation**: Merge data from various sources to create a single table representing each loan application (`SK_ID_CURR`).

## Feature Engineering
1. **Manual Feature Engineering**: Use domain knowledge to create meaningful features from historical data.
2. **Automated Feature Engineering**: Utilize Featuretools to generate extensive features.
3. **Deep Learning-Based**: Extract features using Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).

## Handling Imbalanced Data
- **Undersampling**: Apply hierarchical clustering to balance the dataset by reducing the majority class.
- **Combination**: Use hierarchical clustering to undersample and Synthetic Minority Over-sampling Technique (SMOTE) to increase minority class samples.

## Machine Learning Models
- **Algorithms Used**: 
  - **Boosted Trees**: XGBoost, LightGBM, CatBoost
  - **Fully Connected Neural Network (FCNN)**
- **Handling Missing Data**: 
  - **Boosted Trees**: Naturally handle missing values.
  - **FCNN**: Requires imputation; categorical variables imputed as 'Not Available', numerical features imputed with column mean.
- **Categorical Variables**: 
  - **XGBoost & FCNN**: Use one-hot encoding.
  - **LightGBM & CatBoost**: Handle categorical features natively; encode categories as non-negative integers.

## Performance Metrics
- **Recall**: Focus on reducing false negatives due to the high cost of misclassifying defaulters.
- **Precision & F1-Score**: Provide a balance between precision and recall.
- **Area Under ROC Curve (AUC)**: Assess overall classifier performance.
- **Cohen's Kappa**: Measure agreement between predicted and actual values, emphasizing correct classification.

## Summary
This project involves data collection, preprocessing, feature engineering, and model training to effectively predict loan repayment probabilities. The strategies employed balance the dataset and enhance the feature set to improve model accuracy, particularly in identifying potential defaulters.
