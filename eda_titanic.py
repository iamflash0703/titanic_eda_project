"""
Project 1: Exploratory Data Analysis (EDA) on Titanic Dataset

Goal: Understand the Titanic passenger data, clean it, visualize it,
and find patterns related to survival.
"""

# ---------- STEP 0: Import Libraries ----------
import pandas as pd                     # for data handling (tables/dataframes)
import numpy as np                      # for numerical operations
import matplotlib.pyplot as plt         # for plotting graphs
import seaborn as sns                   # for nicer statistical plots + built-in datasets

# Set a consistent plot style for all charts
sns.set_style("darkgrid")

# ---------- STEP 1: Data Collection ----------
# seaborn ships with a cleaned copy of the Titanic dataset, so we don't
# need to manually download a CSV from Kaggle. This pulls it from
# seaborn's GitHub data repo.
df = sns.load_dataset("titanic")

print("Shape of dataset (rows, columns):", df.shape)
print("\nFirst 5 rows:\n", df.head())

# ---------- STEP 2: Data Cleaning ----------

# 2a. Check missing values in every column
print("\nMissing values per column:\n", df.isnull().sum())

# 2b. 'age' has missing values -> fill with median age (robust to outliers)
df["age"] = df["age"].fillna(df["age"].median())

# 2c. 'embarked' (port of boarding) has a few missing values -> fill with
# the most frequent value (mode), since it's a categorical column
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# 2d. 'deck' column has too many missing values (>75%) -> drop the column
# entirely rather than trying to fill it, since imputing would be unreliable
df = df.drop(columns=["deck"])

# 2e. Drop rows where 'embark_town' is still missing (very few rows affected)
df = df.dropna(subset=["embark_town"])

# 2f. Confirm cleaning worked
print("\nMissing values after cleaning:\n", df.isnull().sum())

# ---------- STEP 3: Summary Statistics ----------
print("\nSummary statistics (numeric columns):\n", df.describe())

# Survival rate overall
survival_rate = df["survived"].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.2f}%")

# Survival rate by gender
survival_by_sex = df.groupby("sex")["survived"].mean() * 100
print("\nSurvival rate by sex:\n", survival_by_sex)

# Survival rate by passenger class
survival_by_class = df.groupby("pclass")["survived"].mean() * 100
print("\nSurvival rate by passenger class:\n", survival_by_class)

# ---------- STEP 4: Visualizations ----------

# 4a. Age distribution histogram
plt.figure(figsize=(7, 5))
sns.histplot(df["age"], bins=30, kde=True, color="#4C9AFF")
plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=150)
plt.close()

# 4b. Survival count by gender (bar plot)
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="sex", hue="survived", palette=["#E74C3C", "#2ECC71"])
plt.title("Survival Count by Gender")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("survival_by_gender.png", dpi=150)
plt.close()

# 4c. Survival rate by passenger class (bar plot)
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="pclass", y="survived", palette="viridis")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_class.png", dpi=150)
plt.close()

# 4d. Fare distribution boxplot (to spot outliers)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="pclass", y="fare", palette="coolwarm")
plt.title("Fare Distribution by Passenger Class (Outlier Check)")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("fare_boxplot.png", dpi=150)
plt.close()

# 4e. Correlation heatmap between numeric features
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

print("\nAll plots saved successfully as PNG files.")

# ---------- STEP 5: Key Insights (for report) ----------
insights = [
    f"Overall survival rate was {survival_rate:.1f}%.",
    f"Female survival rate ({survival_by_sex['female']:.1f}%) was much higher "
    f"than male survival rate ({survival_by_sex['male']:.1f}%).",
    f"1st class passengers had the highest survival rate "
    f"({survival_by_class[1]:.1f}%), while 3rd class had the lowest "
    f"({survival_by_class[3]:.1f}%).",
    "Fare and Passenger class show negative correlation with each other "
    "(higher class number = lower fare, since class 1 is the premium class).",
    "Age was moderately filled using median imputation since ~20% values "
    "were missing.",
]

print("\nKey Insights:")
for i, point in enumerate(insights, 1):
    print(f"{i}. {point}")