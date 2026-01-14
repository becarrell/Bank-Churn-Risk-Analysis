# Bank Churn Risk Analysis  
**End-to-End SQL, Python, Machine Learning, and Power BI Project**

## Overview
This project presents a full end-to-end customer churn analysis using bank customer data from Kaggle.  
The objective is to identify key drivers of churn, segment customers by risk and value, and quantify the potential business impact of targeted retention strategies.

The project is intentionally structured to mirror a real-world analytics workflow:
- Raw data ingestion and cleaning in SQL
- Feature engineering and rule-based logic grounded in exploratory analysis
- Statistical and machine learning modeling in Python
- Executive-facing dashboards in Power BI
- Translation of model outputs into actionable business insights

The primary audience for this analysis is a bank or financial services organization seeking to reduce customer churn and prioritize retention efforts.

NOTE: This is a work in progress and updates will be added at a later date. Future plans include comparing different methods of clustering, SHAP analysis, comparing other methods of ML,
and hyperparameter tuning updates.

---

## Business Problem
Customer churn represents a direct loss of revenue and future customer lifetime value (CLV).  
Rather than reacting after customers leave, banks benefit from identifying **which active customers are most at risk** and **where retention efforts will have the highest financial impact**.

This project answers four core questions:
1. What factors are most strongly associated with customer churn?
2. Which customers should be prioritized for retention?
3. How does a simple rule-based approach compare to a machine learning model?
4. What is the potential financial upside of targeted retention strategies?

---

## Dataset
**Source:** Kaggle – Bank Customer Churn Dataset  

**Observations:** ~10,000 customers  
**Target Variable:** `Exited` (1 = churned, 0 = retained)

### Key Features
- Customer demographics (age, gender, geography)
- Account attributes (balance, tenure, number of products)
- Engagement indicators (activity status, complaints, satisfaction score)
- Financial metrics (estimated salary, credit score, points earned)

---

## Tech Stack
- **PostgreSQL** – data cleaning, feature engineering, exploratory analysis
- **Python** – EDA, visualization, modeling, and business impact analysis  
  - pandas, numpy
  - scikit-learn
  - statsmodels
  - matplotlib, seaborn
- **Power BI** – interactive dashboards and executive summaries
- **GitHub** – project organization and version control

---

## Project Architecture

SQL (Raw → Clean → Features)
↓
Python (EDA → Modeling → Impact)
↓
Power BI (Dashboards & Insights)


---

## SQL: Data Cleaning & Feature Engineering
Raw data was ingested into PostgreSQL and transformed into a clean analytical table.

Key steps included:
- Handling missing values using dataset-level averages or defaults
- Standardizing categorical fields (gender, geography, card type)
- Engineering analytical features:
  - Customer lifetime value (CLV) estimate
  - Age, credit score, balance, and salary buckets
- Designing a **rule-based churn risk classification** grounded in exploratory analysis

The final table serves as the single source for all downstream analysis.

---

## Exploratory Data Analysis (EDA)
EDA was performed using both SQL and Python to understand:
- Differences between churned and retained customers
- Relationships between churn and engagement, satisfaction, and product usage
- Distributional differences across demographic and financial variables

Key findings:
- Customers with fewer products and lower engagement churn at significantly higher rates
- A customer complaint had a near one to one correlation with churn indicators
- Churn risk varies meaningfully across age groups and geographies
- High-value customers (higher balance and CLV) still represent meaningful churn risk

---

## Modeling Approach

### 1. Rule-Based Churn Risk (SQL)
A transparent, interpretable churn risk classification was created using SQL rules derived from EDA.

Customers were categorized as:
- **Low Risk**
- **Medium Risk**
- **High Risk**
- **Churned**

This approach prioritizes interpretability and mirrors how many business teams initially approach churn management.

---

### 2. Logistic Regression
A logistic regression model was built to:
- Quantify the directional impact of key features
- Provide interpretability through coefficients
- Check for multicollinearity using VIF

This model serves as a statistical baseline and interpretability benchmark.

---

### 3. Random Forest Classifier
A Random Forest model was trained to capture non-linear relationships and interactions.

Modeling details:
- Stratified train-test split (70/30)
- Class-weighting to address churn imbalance
- Cross-validation using ROC-AUC
- Feature importance analysis
- Decile-based lift analysis for targeting effectiveness

---

## Model Performance Summary
- **Random Forest outperformed rule-based logic** in ranking churn risk
- ROC-AUC demonstrated strong discriminatory power
- Decile analysis showed substantially higher churn rates in top-risk segments
- Results support using ML outputs to prioritize retention outreach

---

## Customer Segmentation
Unsupervised clustering (DBSCAN) was applied to identify naturally occurring customer segments based on behavior, engagement, and value.

Clustering was used to:
- Explore heterogeneity within churn and non-churn groups
- Support targeted retention strategies
- Provide qualitative insight into customer profiles

---

## Business Impact Analysis
Rather than stopping at model accuracy, the analysis translates predictions into financial impact.

Assumptions:
- Retention success rate: **15%**
- Focus on **high-risk but still active customers**

Outputs:
- Total CLV at risk
- Potential CLV saved under rule-based targeting
- Potential CLV saved using Random Forest targeting
- Incremental benefit of ML-driven prioritization

This framing aligns with how analytics teams justify investment in modeling and retention programs.

---

## Power BI Dashboard
An interactive Power BI dashboard was created to present insights to non-technical stakeholders.

Dashboard pages include:
1. **Overview**
   - Overall churn rate
   - Average CLV
   - Average Balance
   - High risk customers
   - Geographic churn patterns
2. **Targeting & Business Impact**
   - Rule-based vs ML targeting comparison
   - Retention value scenarios

pdf is available in the `powerbi/` folder.

---

## Key Takeaways
- Churn is strongly driven by age, geography, gender, and balance
- Rule-based approaches are interpretable but limited in precision
- Machine learning improves prioritization of high-risk customers
- Framing churn in terms of CLV materially improves decision-making

---

## Limitations & Next Steps
- CLV is estimated using a heuristic rather than transaction-level data
- Retention success rate is assumed rather than empirically measured
- This is synthetic data, for example the credit scores would not apply in these geographic regions
- No time series data. Snap shot data is limited prediction models
- Individual cause for churn is unspecified and makes it difficult to provide business recomendations
- Future work could include:
  - Cost-sensitive optimization
  - Model monitoring and drift analysis
  - A/B testing retention strategies
  - Integration with real-time scoring pipelines
  - SHAP, xgboost, and GMM


---

## How to Use This Repository
1. Review SQL scripts to understand data preparation
2. Run the Python analysis for EDA, modeling, and impact estimation
3. Open the Power BI dashboard for key insights
4. Use this project as a template for future customer analytics work

---

## Author
**Bret Carrell**  
B.S. Economics & Data Analytics  
Indiana University – Kelley School of Business

