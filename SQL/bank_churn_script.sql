-- Project: Bank Customer Churn Analysis & Risk Segmentation
-- Objective: Clean raw data, engineer features, and derive actionable insights for retention strategy

DROP TABLE IF EXISTS bank_churn_clean;

-- Build cleaned table with engineered features
CREATE TABLE bank_churn_clean AS
WITH averages AS (
    SELECT
        AVG(creditscore) AS avg_credit,
        AVG(age) AS avg_age,
        AVG(balance) AS avg_balance,
        AVG(estimatedsalary) AS avg_salary,
        AVG(satisfactionscore) AS avg_satisfaction
    FROM bank_churn_raw
),
cleaned AS (
    SELECT
        r.CustomerId AS customer_id,

        -- Replace NULLs using averages or 0
		COALESCE(r.age, a.avg_age) AS age,
		COALESCE(r.tenure, 0) AS tenure,
		COALESCE(r.numofproducts, 0) AS number_of_products,
		COALESCE(r.hascrcard, 0) AS has_card,
		COALESCE(r.isactivemember, 0) AS is_active,
		COALESCE(r.complain, 0) AS has_complained,
		COALESCE(r.satisfactionscore, a.avg_satisfaction) AS satisfaction_score,
		COALESCE(r.pointsearned, 0) AS points_earned,
        COALESCE(r.creditscore, a.avg_credit) AS credit_score,
        COALESCE(r.balance, a.avg_balance) AS balance,
        COALESCE(r.estimatedsalary, a.avg_salary) AS salary,
        COALESCE(r.exited, 0) AS has_churned,

		-- Gender as 1 or 0
        CASE 
            WHEN LOWER(TRIM(r.gender)) IN ('female','f','woman') THEN 1 
            ELSE 0 
        END AS is_female,

        -- Geography standardized
        CASE 
            WHEN LOWER(TRIM(r.geography)) IN ('fr','france') THEN 'France'
            WHEN LOWER(TRIM(r.geography)) IN ('es','spain') THEN 'Spain'
            WHEN LOWER(TRIM(r.geography)) IN ('de','germany') THEN 'Germany'
            ELSE 'Other'
        END AS geography,

        -- Card type standardized
        CASE 
            WHEN LOWER(TRIM(r.cardtype)) LIKE 'gold%' THEN 'Gold'
            WHEN LOWER(TRIM(r.cardtype)) LIKE 'silver%' THEN 'Silver'
            WHEN LOWER(TRIM(r.cardtype)) LIKE 'diamond%' THEN 'Diamond'
            WHEN LOWER(TRIM(r.cardtype)) LIKE 'platinum%' THEN 'Platinum'
            ELSE 'Other'
        END AS cardtype,

        -- Customer lifetime value estimated
        (
            (COALESCE(r.balance, a.avg_balance) * 0.01) +
            (COALESCE(r.estimatedsalary, a.avg_salary) * 0.003)
        ) * (1 + COALESCE(r.tenure, 0)) AS clv_estimated

    FROM bank_churn_raw r
    CROSS JOIN averages a
)
SELECT 
    *,
	-- Age Bucket
    CASE
        WHEN age = 0 THEN 'Unknown'
        WHEN age > 60 THEN 'Senior'
        WHEN age > 45 THEN 'Middle'
        WHEN age > 20 THEN 'Adult'
        ELSE 'Youth'
    END AS age_bucket,

    -- Credit score bucket
    CASE
        WHEN credit_score < 600 THEN 'Low'
        WHEN credit_score < 750 THEN 'Medium'
        ELSE 'High'
    END AS credit_bucket,

    -- Balance bucket
    CASE
        WHEN balance = 0 THEN 'Zero'
        WHEN balance < 50000 THEN 'Low'
        WHEN balance < 150000 THEN 'Medium'
        ELSE 'High'
    END AS balance_bucket,

    -- Salary bucket
    CASE
        WHEN salary = 0 THEN 'Zero'
        WHEN salary < 50000 THEN 'Low'
        WHEN salary < 150000 THEN 'Medium'
        ELSE 'High'
    END AS salary_bucket,

    -- CLV bucket
    CASE
        WHEN clv_estimated = 0 THEN 'Zero'
        WHEN clv_estimated < 3000 THEN 'Low'
        WHEN clv_estimated < 10000 THEN 'Medium'
        ELSE 'High'
    END AS clv_bucket,

	-- Churn Risk Buckets based on EDA. over 50 % churn rate for 'High Risk' and over 25 % for 'Medium Risk'
	CASE 
	    WHEN has_churned = 1 THEN 'Churned'
	    WHEN has_complained = 1 THEN 'High Risk'
		WHEN number_of_products >= 3 THEN 'High Risk'
		WHEN age > 45 AND age < 60 THEN 'High Risk'
	    WHEN is_active = 0 AND geography = 'Germany' AND balance > 100000 AND balance < 120000 THEN 'High Risk'
	    WHEN is_active = 0 OR age > 40 OR satisfaction_score = 2 AND number_of_products = 1 THEN 'Medium Risk'
		WHEN number_of_products < 2 THEN 'Medium Risk'
		WHEN geography = 'Germany' THEN 'Medium Risk'
	    ELSE 'Low Risk'
	END AS churn_risk

FROM cleaned;

-- Create index
CREATE INDEX IF NOT EXISTS idx_churn_customer_id ON bank_churn_clean(customer_id);
