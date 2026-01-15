
# ===== PHASE I =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("HR_Analytics.csv")
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df = df.dropna()
df.duplicated().sum()
df = df.drop_duplicates()
df.describe(include="object")

for col in df.select_dtypes(include="object").columns:
    print(col)
    print(df[col].value_counts())
    print("-" * 30)

plt.figure(figsize=(8, 6))
sns.countplot(x="Attrition", data=df)
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df["Age"], bins=20, kde=True)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.show()

corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap="coolwarm")
plt.show()


# ===== PHASE II =====
from scipy import stats

# Hypothesized benchmark age
benchmark_age = 35

t_stat, p_value = stats.ttest_1samp(df["Age"], benchmark_age)

print("One-sample t-test (Age vs Benchmark)")
print("t-statistic:", t_stat)
print("p-value:", p_value)

education_groups = [
    group["Age"].values
    for name, group in df.groupby("Education")
]

f_stat, p_value = stats.f_oneway(*education_groups)

print("\nANOVA: Age across Education Levels")
print("F-statistic:", f_stat)
print("p-value:", p_value)

male_js = df[df["Gender"] == "Male"]["JobSatisfaction"]
female_js = df[df["Gender"] == "Female"]["JobSatisfaction"]

t_stat, p_value = stats.ttest_ind(male_js, female_js, equal_var=False)

print("\nGender vs Job Satisfaction")
print("t-statistic:", t_stat)
print("p-value:", p_value)

contingency_table = pd.crosstab(df["Gender"], df["Attrition"])

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\nGender vs Attrition (Chi-square)")
print("Chi2:", chi2)
print("p-value:", p)

plt.figure(figsize=(8, 6))
sns.boxplot(x=df["TrainingTimesLastYear"])
plt.title("Training Times Last Year – Outlier Detection")
plt.show()



plt.figure(figsize=(10, 6))
sns.countplot(x="JobSatisfaction", hue="Attrition", data=df)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x="WorkLifeBalance", hue="Attrition", data=df)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x="OverTime", hue="Attrition", data=df)
plt.show()

attrition_rate = df["Attrition"].value_counts(normalize=True) * 100
attrition_rate

plt.figure(figsize=(10, 6))
sns.barplot(x=attrition_rate.index, y=attrition_rate.values)
plt.ylabel("Percentage")
plt.show()

grouped = df.groupby("Department")["Attrition"].value_counts(normalize=True).unstack()
grouped

grouped.plot(kind="bar", figsize=(12, 6))
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Department", y="MonthlyIncome", hue="Attrition", data=df)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x="Attrition", y="YearsAtCompany", data=df)
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x="Age", y="MonthlyIncome", hue="Attrition", data=df)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x="EducationField", hue="Attrition", data=df)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x="BusinessTravel", hue="Attrition", data=df)
plt.show()


# ===== PHASE III =====

import statsmodels.api as sm


df_model = df.copy()

df_model["Attrition"] = df_model["Attrition"].map({"Yes": 1, "No": 0})

X = df_model[["Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction"]]
y = df_model["Attrition"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

model.summary()

# ===== PHASE IV =====
# Predicting Monthly Income using Experience

X_income = df[["TotalWorkingYears"]]
y_income = df["MonthlyIncome"]

X_income = sm.add_constant(X_income)

income_model = sm.OLS(y_income, X_income).fit()

income_model.summary()

# Correlation between Age and Total Working Years
corr_age_exp = df[["Age", "TotalWorkingYears"]].corr()

print("\nCorrelation between Age and Experience")
print(corr_age_exp)
