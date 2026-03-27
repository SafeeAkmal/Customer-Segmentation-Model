# Data Dictionary — Customer Personality Analysis

Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis  
Rows: ~2,240 | Columns: 29 | Separator: tab (\t) | Encoding: latin-1

---

## People

| Column         | Type   | Description                           |
|----------------|--------|---------------------------------------|
| ID             | int    | Unique customer identifier (DROP)     |
| Year_Birth     | int    | Customer birth year → engineer Age    |
| Education      | string | Education level (Graduation, PhD ...) |
| Marital_Status | string | Relationship status                   |
| Income         | float  | Annual household income (24 nulls)    |
| Kidhome        | int    | Number of children at home            |
| Teenhome       | int    | Number of teenagers at home           |
| Dt_Customer    | date   | Date enrolled → engineer Tenure       |
| Recency        | int    | Days since last purchase              |
| Complain       | int    | 1 if complained in last 2 years       |

---

## Spend (Mnt* columns)

| Column           | Description                     |
|------------------|---------------------------------|
| MntWines         | Spend on wine (last 2 years)    |
| MntFruits        | Spend on fruits                 |
| MntMeatProducts  | Spend on meat                   |
| MntFishProducts  | Spend on fish                   |
| MntSweetProducts | Spend on sweets                 |
| MntGoldProds     | Spend on gold products          |

---

## Purchase Channels (Num* columns)

| Column              | Description                          |
|---------------------|--------------------------------------|
| NumDealsPurchases   | Purchases made with a discount       |
| NumWebPurchases     | Purchases through the website        |
| NumCatalogPurchases | Purchases via catalogue              |
| NumStorePurchases   | Purchases in physical store          |
| NumWebVisitsMonth   | Website visits in the last month     |

---

## Campaign Response

| Column      | Description                                      |
|-------------|--------------------------------------------------|
| AcceptedCmp1| 1 if accepted offer in campaign 1                |
| AcceptedCmp2| 1 if accepted offer in campaign 2                |
| AcceptedCmp3| 1 if accepted offer in campaign 3                |
| AcceptedCmp4| 1 if accepted offer in campaign 4                |
| AcceptedCmp5| 1 if accepted offer in campaign 5                |
| Response    | 1 if accepted offer in last campaign (POST-HOC)  |

---

## To Drop

| Column       | Reason                                      |
|--------------|---------------------------------------------|
| ID           | Arbitrary identifier, no predictive value   |
| Z_CostContact| Constant column, zero variance              |
| Z_Revenue    | Constant column, zero variance              |
| Dt_Customer  | Replaced by Customer_Tenure after engineering|
| Year_Birth   | Replaced by Age after engineering           |

---

## Engineered Features (Sprint 2)

| Column          | Formula                                         |
|-----------------|-------------------------------------------------|
| TotalSpend      | Sum of all Mnt* columns                         |
| Age             | 2024 - Year_Birth                               |
| Customer_Tenure | (reference_date - Dt_Customer).days             |
| Family_Size     | Kidhome + Teenhome + (2 if coupled else 1)      |
| TotalPurchases  | Sum of all Num*Purchases columns                |