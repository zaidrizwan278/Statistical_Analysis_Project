# Statistical Workforce Analysis

This project performs a **statistical analysis of HR workforce data** using Python to uncover key insights into employee demographics, experience, training, and compensation. The goal is to support **data-driven HR decision-making**.

---

## Project Overview

The dataset is **cross-sectional (1 time point)**, so the analysis focuses on **statistical insights** rather than temporal modeling.  
The analysis is conducted in **3 phases**:

1. **Data Cleaning & Validation**  
   - Ensured consistency and prepared data for analysis.

2. **Descriptive Statistics & Hypothesis Testing**  
   - Analyzed distributions, group differences across **education levels** and **gender**.

3. **Inferential & Regression Analysis**  
   - Identified correlations and key predictors of outcomes like **Monthly Income**.

---

## Dataset

Variables included:

- `Age`  
- `Monthly Income`  
- `Total Working Years`  
- `Years at Company`  
- `Training Times Last Year`  
- `Department`  
- `Education`  
- `Gender`  

> ⚠️ *Sensitive or private data should be handled carefully. Only include anonymized or sample datasets if publishing publicly.*

---

## Key Findings

- **Employee age is significantly higher than the benchmark** (p < 0.05) → a more experienced workforce.  
- **Age varies significantly across 5 education levels** (p < 0.05).  
- **Strong positive correlation between Age and Total Working Years**.  
- **Total Working Years is a significant predictor of Monthly Income** (p < 0.01).  
- **No significant gender differences** in job satisfaction or attrition (p > 0.05).  

---

## Tools & Methods

- **Python** 🐍  
- **Libraries:** pandas, numpy, matplotlib, seaborn, statsmodels, scipy  
- **Techniques:**  
  - Descriptive statistics  
  - Hypothesis testing (t-test, ANOVA)  
  - Correlation analysis  
  - Linear regression  

---
