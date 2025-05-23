# Bank Loan Approval Prediction using Machine Learning

This project was developed as part of the Strategic Capstone Project (DSC 789-001) at Adelphi University in Spring 2025. The goal of this project was to leverage machine learning techniques to automate and improve the accuracy of loan approval decisions in the banking sector.

## Project Objective

Banks and financial institutions often rely on manual processes for evaluating loan applications. These methods are time-consuming, inconsistent, and susceptible to human bias. This project aims to build an interpretable and data-driven machine learning model that can predict whether a loan should be approved or not based on applicant information.

The primary objectives were:

- To automate the loan approval decision-making process using historical applicant data.
- To improve prediction accuracy and fairness in loan approvals.
- To identify the most influential features affecting approval decisions.
- To provide actionable recommendations based on model insights.

## Dataset Description

The dataset consists of 32,581 individual loan applications sourced from Kaggle. Each entry contains 12 attributes including both numerical and categorical data, such as:

- Applicant income
- Employment length
- Loan amount
- Interest rate
- Credit history length
- Home ownership
- Loan intent
- Default history
- Loan status (target variable)

The target variable is `loan_status` (1 = approved, 0 = rejected). The dataset was heavily imbalanced with only 22% of applications labeled as approved.

## Data Preparation

Data preprocessing included the following steps:

1. **Cleaning and Deduplication**:
   - Removed 165 duplicate records.
   - Filled missing numerical values with median values.
   - Replaced missing categorical values with 'Unknown'.

2. **Feature Engineering**:
   - Created new variables: `income_to_loan_ratio`, `debt_burden_score`, and `log_income`.
   - Binned continuous variables into categorical risk levels.
   - Created binary flags for high-risk profiles.

3. **Encoding**:
   - Applied Ordinal Encoding for ordered categorical features (e.g., loan_grade).
   - Used One-Hot Encoding for nominal variables (e.g., loan intent, home ownership).

4. **Outlier Treatment**:
   - Used the IQR method to cap extreme values in income, loan amount, and interest rate.

5. **Scaling**:
   - Standardized all numerical features using `StandardScaler`.

## Exploratory Data Analysis (EDA)

Key findings from the EDA included:

- Most applicants are young (20–30 years old).
- Renters showed a higher likelihood of loan approval.
- Loans intended for home improvement were more often approved.
- High income and long credit history were not always correlated with approval.
- Strong predictors of approval included debt-to-income ratio, loan size, and loan purpose.

## Handling Class Imbalance

To address the imbalance in the target variable, we applied the following resampling techniques:

- **SMOTE (Synthetic Minority Oversampling Technique)**: Used to create balanced synthetic examples.
- **ADASYN (Adaptive Synthetic Sampling)**: Focused on generating harder-to-classify minority samples.

Resampling was applied only to the training dataset to prevent data leakage.

## Model Building

Five classification algorithms were implemented:

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- K-Nearest Neighbors (KNN)

Each model was tested using different combinations of feature selection and resampling techniques (SMOTE and ADASYN).

## Feature Selection

To identify the most important variables, we applied:

- **LASSO Regression**: Shrinks less important feature coefficients to zero, improving interpretability.
- **Random Forest Feature Importance**: Ranks features based on their contribution to reducing impurity in decision trees.

Consistently important features across models included:

- `loan_percent_income`
- `income_to_loan_ratio`
- `loan_size_cat_Small`
- `debt_burden_score`
- `home_ownership_RENT`

## Evaluation and Results

Performance was measured using the following metrics:

- AUC (Area Under ROC Curve)
- G-Mean (Geometric Mean of sensitivity and specificity)
- Precision and Recall

The best model was:

- **XGBoost with SMOTE and LASSO**
  - AUC: 0.91
  - G-Mean: 0.863

This configuration offered the best balance of accuracy and interpretability.

## Business Recommendations

Based on the model insights, we proposed the following strategies for banks:

- Promote small loans and moderate exposure levels.
- Target renters with flexible products.
- Use income-to-loan ratios in prequalification checks.
- Screen high-income applicants more cautiously.
- Provide tailored offers to homeowners and older applicants.
- Encourage low-risk borrowers with faster processing or incentives.

## Technologies Used

- Python (Pandas, scikit-learn, XGBoost)
- Google Colab
- Microsoft Excel, Word, PowerPoint


## Team Members

- Yamika Ratna Kadiyala
- Roja Eslavath
- Amarnath Dasari
- Suhitha Yalamanchili
- Han Ru Wu

## Acknowledgements

This project was developed under the guidance of Prof. Zahra Sedighi Maman as part of the Strategic Capstone Project course (DSC 789-001) at Adelphi University, Spring 2025.

## References
- Google Colab, OpenAI ChatGPT, Microsoft Office Suite, Kaggle

## Final Code 
- https://drive.google.com/drive/folders/1C4kg-4aymbFeJ64N-RR5TW0Bg7OhFILU?usp=drive_link


