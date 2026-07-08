# 🚢 Titanic Dataset — Exploratory Data Analysis

Exploratory Data Analysis (EDA) on the Titanic dataset to understand survival patterns based on gender, passenger class, age, and fare. 

## 📌 Overview
This project covers the full EDA workflow — data cleaning, summary statistics, and visualization — using Python's Pandas, NumPy, Matplotlib, and Seaborn.

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- Matplotlib, Seaborn

## 🔍 Steps Followed
1. **Data Collection** — Loaded Titanic dataset (891 records) via Seaborn.
2. **Data Cleaning** — Handled missing values in `age` (median imputation), `embarked` (mode imputation), and dropped `deck` column (75%+ missing).
3. **Summary Statistics** — Computed survival rate, grouped stats by gender and class.
4. **Visualization** — Created 5 plots: age distribution, survival by gender, survival by class, fare boxplot, correlation heatmap.
5. **Insights** — Derived key patterns from the data.

## 📊 Key Insights
- Overall survival rate: **38.25%**
- Female survival rate (**74.04%**) was ~4x higher than male (**18.89%**)
- 1st class passengers had the highest survival rate (**62.62%**) vs 3rd class (**24.24%**)
- Fare and passenger class show a negative correlation

## 📁 Files
- `eda_titanic.py` — full commented Python script
- `*.png` — generated visualizations
