# Databricks notebook source
# MAGIC %md
# MAGIC ##  Creating base table for Dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC Total number of Customer Base

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT CustomerID) AS Customers FROM customer_info.online_retail_train

# COMMAND ----------

# MAGIC %md
# MAGIC Customer base by number of years

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT year(invdate), COUNT(DISTINCT CustomerID) AS Customers FROM customer_info.online_retail_train GROUP BY year(invdate) -- Parameterize the year(invdate)

# COMMAND ----------

# MAGIC %md
# MAGIC Calculating Monthwise Customer base distribution

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT month(invDate) AS month,
# MAGIC        -- year(invdate) as year, 
# MAGIC        COUNT(DISTINCT CustomerID) AS Customers,
# MAGIC        COUNT(invoiceno) AS invoices,
# MAGIC        SUM(profit_value) profit
# MAGIC FROM customer_info.online_retail_train
# MAGIC WHERE year(invdate) = 2011 -- parameterize it on Query Window
# MAGIC GROUP BY month(invDate),year(invdate)
# MAGIC ORDER BY --year(invdate),
# MAGIC          month(invDate)

# COMMAND ----------

# MAGIC %md 
# MAGIC Country wise Layout

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT country,
# MAGIC        COUNT(DISTINCT CustomerID) as Customers
# MAGIC FROM customer_info.online_retail_train
# MAGIC WHERE year(invdate) = 2011
# MAGIC GROUP BY country
# MAGIC ORDER BY count(DISTINCT CustomerID) DESC

# COMMAND ----------

# MAGIC %md
# MAGIC Customer Pred Visits

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT COUNT(*) AS Total_Customer, pred_visit FROM 
# MAGIC (
# MAGIC   SELECT CASE WHEN pred_visits <= 10 THEN '< 10 weeks' 
# MAGIC               WHEN pred_visits >= 40 THEN '40 weeks'
# MAGIC               WHEN pred_visits >= 30 AND pred_visits <= 39 THEN '30 - 39 weeks'
# MAGIC               WHEN pred_visits >= 20 AND pred_visits <= 29 THEN '20 - 29 weeks'
# MAGIC               WHEN pred_visits >= 10 AND pred_visits <= 19 THEN '10 - 19 weeks'
# MAGIC          ELSE 'others'
# MAGIC          END AS pred_visit, PRED_CLV
# MAGIC   FROM customer_info.ltv_results
# MAGIC )
# MAGIC GROUP BY pred_visit

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS Total_Customer, Age FROM 
# MAGIC (
# MAGIC   SELECT CASE WHEN Age <= 10 THEN '< 10 days' 
# MAGIC               WHEN Age >= 40 THEN '40 days'
# MAGIC               WHEN Age >= 30 AND Age <= 39 THEN '30 - 39 days'
# MAGIC               WHEN Age >= 20 AND Age <= 29 THEN '20 - 29 days'
# MAGIC               WHEN Age >= 10 AND Age <= 19 THEN '10 - 19 days'
# MAGIC          ELSE 'others'
# MAGIC          END AS age
# MAGIC   FROM customer_info.ltv_results
# MAGIC )
# MAGIC GROUP BY age