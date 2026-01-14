-- Select statements for EDA on the newly created table

-- Overall snapshot of customer churn rate, CLV, satisfaction, and risk distribution.
SELECT 
    COUNT(*) AS total_customers,
    SUM(has_churned) AS total_churned,
    ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS overall_churn_rate,
    ROUND(AVG(clv_estimated), 2) AS avg_clv,
	ROUND(AVG(age), 2) AS avg_age,
    ROUND(AVG(salary), 2) AS avg_salary,
    ROUND(SUM(clv_estimated), 2) AS total_clv,
    SUM(CASE WHEN churn_risk = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_customers,
    ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction
FROM bank_churn_clean;

-- Shows churn rate and CLV differences across regions and gender groups.
SELECT 
    geography,
    CASE WHEN is_female = 1 THEN 'Female' ELSE 'Male' END AS gender,
    COUNT(*) AS customer_count,
    ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS churn_rate,
    ROUND(AVG(clv_estimated), 2) AS avg_clv
FROM bank_churn_clean
GROUP BY geography, is_female
ORDER BY geography, is_female DESC;

-- Correlation for variables on customer churn.
SELECT
    CORR(credit_score, has_churned)::DECIMAL(4,3) AS corr_credit_churn,
    CORR(age, has_churned)::DECIMAL(4,3) AS corr_age_churn,
    CORR(balance, has_churned)::DECIMAL(4,3) AS corr_balance_churn,
    CORR(satisfaction_score, has_churned)::DECIMAL(4,3) AS corr_satisfaction_churn,
    CORR(is_active, has_churned)::DECIMAL(4,3) AS corr_active_churn,
    CORR(has_complained, has_churned)::DECIMAL(4,3) AS corr_complaint_churn,
    CORR(tenure, has_churned)::DECIMAL(4,3) AS corr_tenure_churn,
	CORR(has_card, has_churned)::DECIMAL(4,3) AS corr_has_card_churn,
	CORR(is_female, has_churned)::DECIMAL(4,3) AS corr_gender_churn,
	CORR(points_earned, has_churned)::DECIMAL(4,3) AS corr_points_churn,
	CORR(number_of_products, has_churned)::DECIMAL(4,3) AS corr_products_churn,
	CORR(salary, has_churned)::DECIMAL(4,3) AS corr_salary_churn
FROM bank_churn_clean;

-- The following statements analyze churn rates to determine the churn risk buckets:
-- 1.
SELECT 
    number_of_products,
    COUNT(*) AS customer_count,
    SUM(has_churned) AS churned_customers,
    ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS churn_rate,
    ROUND(AVG(clv_estimated), 2) AS avg_clv
FROM bank_churn_clean
GROUP BY number_of_products
ORDER BY number_of_products;

-- 2.
SELECT 
    age_bucket,
    COUNT(*) AS customer_count,
    SUM(has_churned) AS churned_customers,
    ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS churn_rate
FROM bank_churn_clean
GROUP BY age_bucket
ORDER BY age_bucket;

-- 3.
SELECT 
    has_complained,
    COUNT(*) AS customer_count,
    SUM(has_churned) AS churned_customers,
    ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS churn_rate
FROM bank_churn_clean
GROUP BY has_complained
ORDER BY has_complained;

-- 4.
SELECT 
    geography,
    COUNT(*) AS customer_count,
    SUM(has_churned) AS churned_customers,
    ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS churn_rate
FROM bank_churn_clean
WHERE is_active = 0 AND balance > 100000 AND balance < 120000
GROUP BY geography
ORDER BY geography;

-- 5.
SELECT
	satisfaction_score,
    COUNT(*) AS customer_count,
    SUM(has_churned) AS churned_customers,
    ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS churn_rate
FROM bank_churn_clean
WHERE is_active = 0 OR age > 40 OR satisfaction_score = 2 AND number_of_products < 2
GROUP BY satisfaction_score
ORDER BY satisfaction_score;

-- 6.
SELECT 
    COUNT(*) AS customer_count,
    SUM(has_churned) AS churned_customers,
    ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS churn_rate
FROM bank_churn_clean
WHERE satisfaction_score = 2 and number_of_products = 1

-- Potential impact of targeting High Risk non-churned customers
SELECT
    'High Risk - Still Active' AS segment,
    COUNT(*) AS customers,
    ROUND(AVG(clv_estimated), 2) AS avg_clv,
    ROUND(SUM(clv_estimated), 0) AS total_clv_at_risk,
    ROUND(100.0 * SUM(clv_estimated) / (SELECT SUM(clv_estimated) FROM bank_churn_clean), 2) AS pct_of_total_clv
FROM bank_churn_clean
WHERE churn_risk = 'High Risk' AND has_churned = 0;

-- Profiles customers in each CLV tier by demographics, churn rate, and risk level.
SELECT 
    clv_bucket,
    COUNT(*) AS total_customers,
    ROUND(AVG(age), 2) AS avg_age,
    ROUND(AVG(credit_score), 2) AS avg_credit,
    ROUND(AVG(balance), 2) AS avg_balance,
    ROUND(AVG(salary), 2) AS avg_salary,
    ROUND(AVG(clv_estimated), 2) AS avg_clv,
    ROUND(AVG(tenure), 2) AS avg_tenure,
    ROUND(AVG(is_active), 2) AS avg_is_active,
    ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction,
	ROUND(100.0 * SUM(has_churned) / COUNT(*), 2) AS churn_rate,
	SUM(CASE WHEN churn_risk = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_customers
FROM bank_churn_clean
GROUP BY clv_bucket
ORDER BY 
    CASE clv_bucket
        WHEN 'Zero' THEN 1
        WHEN 'Low' THEN 2
        WHEN 'Medium' THEN 3
        WHEN 'High' THEN 4
    END;

-- Summarizes customer characteristics across churn risk categories.
SELECT 
    churn_risk,
    COUNT(*) AS total_customers,
    ROUND(AVG(age), 2) AS avg_age,
    ROUND(AVG(credit_score), 2) AS avg_credit,
    ROUND(AVG(balance), 2) AS avg_balance,
    ROUND(AVG(salary), 2) AS avg_salary,
    ROUND(AVG(clv_estimated), 2) AS avg_clv,
    ROUND(AVG(tenure), 2) AS avg_tenure,
    ROUND(AVG(is_active), 2) AS avg_is_active,
    ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction
FROM bank_churn_clean
GROUP BY churn_risk
ORDER BY 
    CASE churn_risk
        WHEN 'Churned' THEN 1
        WHEN 'High Risk' THEN 2
        WHEN 'Medium Risk' THEN 3
        WHEN 'Low Risk' THEN 4
    END;