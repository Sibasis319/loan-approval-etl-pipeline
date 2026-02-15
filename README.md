# loan-approval-etl-pipeline
Designed and implemented a batch ETL pipeline to ingest, clean, transform, and analyze loan application data for predictive analytics using Python and Pandas.

# 🏦 Loan Approval Analytics & Data Pipeline Project

## 📌 Project Overview

This project focuses on building a complete data processing and analytics pipeline for loan approval prediction.

Instead of only training a machine learning model, this project emphasizes:

- Data Cleaning & Transformation
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Pipeline Design
- Business Insight Generation
- Predictive Modeling

This demonstrates skills relevant to both **Data Analyst** and **Data Engineer** roles.

---

# 🎯 Business Problem

Financial institutions must evaluate loan applications efficiently while minimizing risk.

This project aims to:

- Analyze key drivers of loan approval
- Build a structured data preprocessing pipeline
- Generate actionable insights from structured data
- Develop a predictive classification model

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📊 Data Engineering Components

## 1️⃣ Data Ingestion
- Imported structured CSV datasets
- Validated schema and column data types

## 2️⃣ Data Cleaning
- Handled missing values using:
  - Mode (Categorical Features)
  - Median (Loan Amount)
  - Mean (Where applicable)
- Standardized inconsistent categorical values
- Converted string categories to numeric format

## 3️⃣ Feature Engineering
- Created Total Income feature
- Applied Log Transformation to reduce skewness
- Performed One-Hot Encoding
- Created categorical income bins
- Correlation matrix generation

## 4️⃣ Data Transformation Pipeline
- Structured preprocessing steps before modeling
- Maintained consistent transformation for train and test datasets
- Column alignment after encoding

---

# 📈 Data Analytics & Insights

## Key Observations:

- Credit History is the strongest predictor of loan approval.
- Applicants with higher total income have higher approval rates.
- Education level slightly influences approval probability.
- Loan Amount distribution is right-skewed (log transformation applied).

Visualizations included:
- Distribution Plots
- Boxplots
- Stacked Bar Charts
- Correlation Heatmap

---

# 🤖 Predictive Modeling

## Model Used:
Logistic Regression

## Evaluation Metrics:

| Metric        | Score |
|--------------|--------|
| Accuracy     | 83% |
| Precision    | 0.83 |
| Recall       | 0.94 |
| F1-Score     | 0.88 |

The model performs well in identifying approved loans with high recall.

---

# 🏗️ Project Architecture Flow

```
Raw CSV Data
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
EDA & Visualization
      ↓
Encoding & Transformation
      ↓
Model Training
      ↓
Prediction Output (Submission File)
```

---

# 📁 Project Structure

```
Loan-Analytics-Project/
│
├── train_loan.csv
├── test_loan.csv
├── loan_prediction.ipynb
├── logistic.csv
└── README.md
```

---

# 📌 Skills Demonstrated

## Data Engineering Skills
- Data preprocessing pipeline design
- Handling missing and inconsistent data
- Feature transformation
- Schema alignment between datasets
- Efficient dataframe manipulation

## Data Analytics Skills
- Exploratory Data Analysis
- Business insight extraction
- Statistical summary analysis
- Visualization interpretation

## Machine Learning Skills
- Classification modeling
- Model evaluation
- Performance metric interpretation

---

# 🔮 Future Enhancements

- Build automated preprocessing pipeline using Scikit-learn Pipeline
- Store data in SQL database
- Perform SQL-based analysis
- Deploy model as API
- Implement Random Forest / XGBoost
- Add ROC-AUC & Confusion Matrix visualization

---

# 🎯 Conclusion

This project demonstrates the ability to:

- Process raw structured data
- Build a reproducible data pipeline
- Generate business insights
- Develop a predictive classification model
- Deliver production-ready prediction output

It reflects practical skills required for:
- Data Analyst roles
- Data Engineering internships
- Entry-level ML positions

---

# 👨‍💻 Author

Sibasis Sahu  
B.Tech Student  
Aspiring Data Engineer | Data Analyst  
