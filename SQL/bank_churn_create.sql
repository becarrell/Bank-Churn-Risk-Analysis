-- Importing data from Kaggle csv

DROP TABLE IF EXISTS bank_churn_raw;

CREATE TABLE bank_churn_raw (
    RowNumber INT,
    CustomerId BIGINT,
    Surname VARCHAR(50),
    CreditScore INT,
    Geography VARCHAR(20),
    Gender VARCHAR(10),
    Age INT,
    Tenure INT,
    Balance NUMERIC(15,2),
    NumOfProducts INT,
    HasCrCard INT,
    IsActiveMember INT,
    EstimatedSalary NUMERIC(15,2),
    Exited INT,
    Complain INT,
    SatisfactionScore INT,
    CardType VARCHAR(20),
    PointsEarned INT
);
