# Databricks notebook source
# MAGIC %pip install lifetimes

# COMMAND ----------

# MAGIC %md
# MAGIC # Customer Lifetime Value Forcasting
# MAGIC 
# MAGIC The objective of this notebook is to illustrate how we might calculate the customers monetary value in an efficient manner leveraging the distributed computational power of Databricks and its Delta lake. For this exercise, we will use a popular library called lifetimes (https://pypi.org/project/Lifetimes/)
# MAGIC 
# MAGIC The illustrated use case is non-contractual and Continuous purchases which is the most common and and complex one.
# MAGIC 
# MAGIC ## Introduction
# MAGIC 
# MAGIC Customer lifetime value models (CLVs) are powerful predictive models that allow analysts and data scientists to forecast how much customers are worth to a business. 
# MAGIC 
# MAGIC CLV is a core task and accurately understanding it helps enterprises better target their marketing and retention programs, optimize customer service and business forecasts, and drive higher revenues and improved margins. 
# MAGIC 
# MAGIC Two global retailers leveraged Databricks scalability and ML capabilities to implement their CLV solutions. 
# MAGIC We will share details of the process and lessons learned. 
# MAGIC 
# MAGIC ### Contractual vs Non Contractual
# MAGIC 
# MAGIC * Customer Death can be observed vs Un observed
# MAGIC * Discrete vs Continuous purchases 
# MAGIC * purchases happened occurs at fixed time
# MAGIC * Continuous purchases can happen at any time
# MAGIC 
# MAGIC ### Dataset
# MAGIC 
# MAGIC For our dataset, we will make use of E-commerce data from Kaggle, a tansformed copy of the data can be downloaded from https://github.com/bobeidat/data.git
# MAGIC 
# MAGIC With the dataset accessible within Databricks, let's load it and transform it in the required structre for medling
# MAGIC 
# MAGIC ### Inputs to CLV Models
# MAGIC 
# MAGIC The inputs to the simplest CLV models, e.g., Fader/Hardie, include 
# MAGIC 
# MAGIC * how often each customer makes purchases, 
# MAGIC * how many purchases they make, 
# MAGIC * how long they've been customrs
# MAGIC * how much they spend on average.
# MAGIC 
# MAGIC Data included in the data set:
# MAGIC 
# MAGIC - **`frequency`** represents the number of *repeat* purchases the customer has made. It's the count of time periods the customer had a purchase in. So if using weeks as units, then it's the count of weeks during which the customer had a purchase.   
# MAGIC - **`T`** represents the age of the customer in whatever time units chosen (weekly, in the above dataset). This is equal to the duration between a customer's first purchase and the end of the period under study.
# MAGIC - **`recency`** represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer's first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.)
# MAGIC 
# MAGIC ### Output from CLV Models
# MAGIC 
# MAGIC The basic output from these models include
# MAGIC 
# MAGIC * the probability that a customer will shop in the next period
# MAGIC * the expected number of purchases a customer will make in the next period
# MAGIC 
# MAGIC These two are combined with the customer's average sales or margin to provide the expected customer lifetime value.
# MAGIC 
# MAGIC ### Model Types Used
# MAGIC 
# MAGIC * The Fader/Hardie models build up prior distributions of the customers' behavior based on the data provided which are then used to provide the two outputs.
# MAGIC * Other machine learning based approaches build separate models (but on the same data) to predict the two outputs. For example, to predict the probability a customer hasn't lapsed we could use simple logistic regression. To predict the number of visits we could use a simple linear regression. These are basic models we can use to baseline performance

# COMMAND ----------

train = spark.read.table("customer_info.online_retail_train").cache()
train.createOrReplaceTempView("train")

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESCRIBE train

# COMMAND ----------

# MAGIC %md
# MAGIC The data is for 2011 and it seems Nov has the highest sales value, there is a correlation between 
# MAGIC number of unique customers, profit and invoices

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT month(invDate) AS month,
# MAGIC        -- year(invdate) as year, 
# MAGIC        COUNT(DISTINCT CustomerID) AS Customers,
# MAGIC        COUNT(invoiceno) AS invoices,
# MAGIC        SUM(profit_value) profit
# MAGIC FROM train
# MAGIC WHERE year(invdate) = 2011
# MAGIC GROUP BY month(invDate),year(invdate)
# MAGIC ORDER BY --year(invdate),
# MAGIC          month(invDate)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT country,
# MAGIC        COUNT(DISTINCT CustomerID) as Customers
# MAGIC FROM train
# MAGIC WHERE year(invdate) = 2011
# MAGIC GROUP BY country
# MAGIC ORDER BY count(DISTINCT CustomerID) DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prep Data for Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC As indicated earleir, the lifetime model requires the data to be in a specific structure. 
# MAGIC 
# MAGIC So we create a view to generate T, Recency and Frequency
# MAGIC - **`frequency`** represents the number of *repeat* purchases the customer has made. It's the count of time periods the customer had a purchase in. So if using weeks as units, then it's the count of weeks during which the customer had a purchase.   
# MAGIC - **`T`** represents the age of the customer in whatever time units chosen (weekly, in the above dataset). This is equal to the duration between a customer's first purchase and the end of the period under study.
# MAGIC - **`recency`** represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer's first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW sales_summmary AS
# MAGIC   SELECT CustomerID, 51 - min(weekno) AS T,
# MAGIC          max(weekno) - min(weekno) AS Recency,
# MAGIC          count(*) - 1 AS Frequency, 
# MAGIC          AVG(profit_value) AS profit
# MAGIC   FROM train
# MAGIC   GROUP BY CustomerID

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM sales_summmary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecasting the expected number of visits 

# COMMAND ----------

import pandas as pd
import time as timemodule
from datetime import datetime 
from pyspark.sql.functions import udf, lit, monotonically_increasing_id
from pyspark.sql.types import BooleanType
from lifetimes import BetaGeoFitter, GammaGammaFitter
#spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

summaryDF = spark.read.format('delta').load('dbfs:/user/hive/warehouse/customer_info.db/summary_2011')

# COMMAND ----------

# MAGIC %md Perform a transformation so the data type is float

# COMMAND ----------

#summaryDF.recency.cast("float")
#summaryDF.T.cast("float")
#summaryDF=summaryDF.repartition(1).cache()
# This will return a new DF with all the columns + id
summaryDF = summaryDF.withColumn("RECENCY", summaryDF.recency1.cast("float"))
summaryDF = summaryDF.withColumn("AGE", summaryDF.T1.cast("float"))
summaryDF = summaryDF.withColumn("ID", monotonically_increasing_id()+1)
summaryDF = summaryDF.withColumn("AVG_MONETARY_VALUE", summaryDF.profit.cast("float")) 
summaryDF = summaryDF.drop("profit", "T1", "recency1")
display(summaryDF)

# COMMAND ----------

data = summaryDF.toPandas().set_index('ID')
data.head()

# COMMAND ----------

# Change datatypes of columns - required to feed into model. 
for col in ['RECENCY', 'AGE', 'AVG_MONETARY_VALUE']:
  data[col] = data[col].astype('float')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training on the BG/NBD model
# MAGIC 
# MAGIC (http://mktg.uni-svishtov.bg/ivm/resources/Counting_Your_Customers.pdf)

# COMMAND ----------

# Fitting beta geo fitter and predicting the frequency and alive probability
bgf_actual = BetaGeoFitter(penalizer_coef=0.0001)
bgf_actual.fit(data['FREQUENCY'], data['RECENCY'], data['AGE'])
bgf_actual.summary

# COMMAND ----------

# MAGIC %md Specify two required parameters :
# MAGIC * t as number of the future time units days/weeks
# MAGIC * time number of the future months

# COMMAND ----------

t = 52.08 # 365 days
time = 12.0

# COMMAND ----------

# MAGIC %md Predict future visists 

# COMMAND ----------

data['PRED_VISITS'] = bgf_actual.conditional_expected_number_of_purchases_up_to_time(t, data['FREQUENCY'], data['RECENCY'], data['AGE'])
data.sort_values(by=['PRED_VISITS']).head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Highlights 
# MAGIC ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) 
# MAGIC 
# MAGIC * Predicted visits 
# MAGIC * Calculated per time units so pred_visits = 5 means 5 (weeks) and it could be any number of visits in any week
# MAGIC * This information can be used for Ranking customers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probability of still being alive
# MAGIC Another interesting metric to look at is the probability of still being alive, let us list our customers based on the probability of being alive 

# COMMAND ----------

# Fitting beta geo fitter and predicting the frequency and alive probability
bgf_actual = BetaGeoFitter(penalizer_coef=0.0001)
bgf_actual.fit(data['FREQUENCY'], data['RECENCY'], data['AGE'])

data['PRED_VISITS'] = bgf_actual.conditional_expected_number_of_purchases_up_to_time(t, data['FREQUENCY'], data['RECENCY'], data['AGE'])
# Compute the probability that a customer with history (frequency, recency, T) is currently alive
data['PROB_ALIVE'] = bgf_actual.conditional_probability_alive(data['FREQUENCY'], data['RECENCY'], data['AGE'])
data.head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC The probability of still being alive in 10 weeks

# COMMAND ----------

# conditional_probability_alive(frequency, recency, T)
# Compute the probability that a customer with history (frequency, recency, T) is currently alive.

data['PROB_ALIVE_t'] = bgf_actual.conditional_probability_alive(10, data['FREQUENCY'], data['RECENCY'])
#Conditional probability alive.
#Conditional probability customer is alive at transaction opportunity n_periods in_future
data.head(10)

# COMMAND ----------

#individual prediction
t = 10 #predict purchases in 10 periods
individual = data.iloc[20]
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time`
bgf_actual.predict(t, individual['FREQUENCY'], individual['RECENCY'], individual['AGE'])

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Estimating customer lifetime value using the Gamma-Gamma model

# COMMAND ----------

from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value

# Fitting gamma gamma fitter and predicting the ltv score
refined_data = data[data['FREQUENCY'] > 1]
refined_data

# COMMAND ----------

# MAGIC %md 
# MAGIC The Gamma-Gamma submodel, in fact, assumes that there is no relationship between the avg monetary value and the purchase frequency. 
# MAGIC In practice we need to check whether the Pearson correlation between the two vectors is close to 0 in order to use this model

# COMMAND ----------

refined_data[['AVG_MONETARY_VALUE', 'FREQUENCY']].corr()

# COMMAND ----------

# MAGIC %md Training on the GammaGammaFitter

# COMMAND ----------

from lifetimes import GammaGammaFitter

ggf_actual = GammaGammaFitter(penalizer_coef = 0.0001)
ggf_actual.fit(refined_data['FREQUENCY'], refined_data['AVG_MONETARY_VALUE'])
ggf_actual.summary

# COMMAND ----------

# freq='W' to tell customer_lifetime_value that the bgf model was trained on weeks / frequency
# PRED_CLV estimate the average transaction value
data['PRED_CLV'] = ggf_actual.customer_lifetime_value(
    bgf_actual,
    data['FREQUENCY'],
    data['RECENCY'],
    data['AGE'],
    data['AVG_MONETARY_VALUE'],
    freq='W',
    time=time, # months
    discount_rate=0.0056  # monthly discount rate 
)

data['COND_EXP_AVG_PROFT'] = ggf_actual.conditional_expected_average_profit(
    data['FREQUENCY'], 
    data['AVG_MONETARY_VALUE']
) 

# estimate the average transaction value
data['COND_EXP_AVG_PROFT'] = ggf_actual.conditional_expected_average_profit(
    data['FREQUENCY'], 
    data['AVG_MONETARY_VALUE']
)
# so you can compare the previous value to data['AVG_MONETARY_VALUE'].mean()

# COMMAND ----------

# Save LTV Results
df_ltv_results = spark.createDataFrame(data)
#spark.sql("DROP TABLE IF EXISTS retail.ltv_results_2011")
#df_ltv_results.write.mode('overwrite').format("delta").save("/tmp/KnowledgeRepo/ML/cust_lifetime_value/delta/ltv_results_2011")
#spark.sql("CREATE TABLE retail.ltv_results_2011 USING DELTA LOCATION '/tmp/KnowledgeRepo/ML/cust_lifetime_value/delta/ltv_results_2011'")

# COMMAND ----------

df_ltv_results.createOrReplaceTempView('LTVData')

# COMMAND ----------

df_ltv_results.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interesting customers segmentations and ranking

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS customer_info.ltv_results;
# MAGIC CREATE TABLE customer_info.ltv_results AS
# MAGIC SELECT * FROM LTVdata;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT customerID, Frequency, Recency, AGE
# MAGIC FROM LTVdata 
# MAGIC ORDER BY PRED_CLV DESC

# COMMAND ----------

# MAGIC %md
# MAGIC Generating the customers segments based on the predectid visits

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT COUNT(*), pred_visit FROM 
# MAGIC (
# MAGIC   SELECT CASE WHEN pred_visits <= 10 THEN '< 10 weeks' 
# MAGIC               WHEN pred_visits >= 40 THEN '40 weeks'
# MAGIC               WHEN pred_visits >= 30 AND pred_visits <= 39 THEN '30 - 39 weeks'
# MAGIC               WHEN pred_visits >= 20 AND pred_visits <= 29 THEN '20 - 29 weeks'
# MAGIC               WHEN pred_visits >= 10 AND pred_visits <= 19 THEN '10 - 19 weeks'
# MAGIC          ELSE 'others'
# MAGIC          END AS pred_visit, PRED_CLV
# MAGIC   FROM LTVdata
# MAGIC )
# MAGIC GROUP BY pred_visit

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT COUNT(*) AS count, pred_visit, SUM(PRED_CLV) AS LTV FROM 
# MAGIC (
# MAGIC   SELECT CASE WHEN pred_visits <= 10 THEN '< 10 weeks' 
# MAGIC               WHEN pred_visits >= 40 THEN '40 weeks'
# MAGIC               WHEN pred_visits >= 30 AND pred_visits <= 39 THEN '30 - 39 weeks'
# MAGIC               WHEN pred_visits >= 20 AND pred_visits <= 29 THEN '20 - 29 weeks'
# MAGIC               WHEN pred_visits >= 10 AND pred_visits <= 19 THEN '10 - 19 weeks'
# MAGIC          ELSE 'others'
# MAGIC          END AS pred_visit, PRED_CLV
# MAGIC   FROM LTVdata
# MAGIC )
# MAGIC GROUP BY pred_visit

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Scaling with Databricks using grouped map pandas UDF
# MAGIC 
# MAGIC (https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html)
# MAGIC 
# MAGIC Grouped map Pandas UDFs first splits a Spark DataFrame into groups based on the conditions specified in the groupby operator, applies a user-defined 
# MAGIC function (pandas.DataFrame -> pandas.DataFrame) to each group, combines and returns the results as a new Spark DataFrame.
# MAGIC 
# MAGIC Grouped map Pandas UDFs uses the same function decorator pandas_udf as scalar Pandas UDFs, but they have a few differences. The input and output are pandas.DataFrame, the grouping semantics is defined by “groupby” clause. in regard to the output it can while the return it can be any any size and it is of a StructType that specifies each column name and type of the returned pandas.DataFrame. the transformation from Spark to pandadataframe is implicit so there is no need to code that.
# MAGIC 
# MAGIC The scaling process is mainly about partitioing the data into smaller grou of customer, perform perdiction 
# MAGIC on the partition level and as finally collect the results back. Technically it includes the following steps:
# MAGIC 
# MAGIC * Generate a new column to be used as a partitioning key, it is simply a number 1-20   
# MAGIC * Create the schema of the returned dataframe
# MAGIC * Creeate a function to perform prediction on a partition level
# MAGIC * Use the dataframe method groupBy apply to distribute the the prediction process

# COMMAND ----------

# MAGIC %md
# MAGIC Generate Groupkey as a partitioning key

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW sales_summmary AS
# MAGIC   SELECT CustomerID%1+1 AS GroupKey,
# MAGIC          CustomerID, 51 - min(weekno) AS T,
# MAGIC          max(weekno) - min(weekno) AS Recency,
# MAGIC          count(*) - 1 AS Frequency, 
# MAGIC          SUM(profit_value) AS profit
# MAGIC   FROM train
# MAGIC   GROUP BY CustomerID

# COMMAND ----------

# MAGIC %md Specify the returned schema

# COMMAND ----------

from pyspark.sql.types import *
resultSchema =StructType([StructField('GroupKey',IntegerType(),True),\
                          StructField('CustomerID',IntegerType(),True),\
                          StructField('FREQUENCY',LongType(),True),\
                          StructField('RECENCY',FloatType(),True),\
                          StructField('AGE',FloatType(),True),\
                          StructField('AVG_MONETARY_VALUE',FloatType(),True),\
                          StructField('PRED_VISITS',FloatType(),True), \
                          StructField('PROB_ALIVE',FloatType(),True),\
                          StructField('PROB_ALIVE_t',FloatType(),True), \
                          StructField('PRED_CLV',FloatType(),True)
                          ])

# COMMAND ----------

# MAGIC %md Create a function to perform prediction on a partition level

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

@pandas_udf(resultSchema, PandasUDFType.GROUPED_MAP)
def CLV_CustomerGroup(df):
  # Change datatypes of columns - required to feed into model. 
  for col in ['RECENCY', 'AGE','AVG_MONETARY_VALUE']:
    df[col] = df[col].astype('float')
  # Fitting beta geo fitter and predicting the frequency and alive probability
  bgf_actual = BetaGeoFitter(penalizer_coef=0.0001)
  bgf_actual.fit(df['FREQUENCY'],df['RECENCY'],df['AGE'])
  df['PRED_VISITS'] = bgf_actual.conditional_expected_number_of_purchases_up_to_time(t, df['FREQUENCY'], df['RECENCY'], df['AGE'])

  ## Gamma 
  refined_df = df[df['FREQUENCY'] > 1]
  ggf_actual = GammaGammaFitter(penalizer_coef = 0.01)
  ggf_actual.fit( refined_df['FREQUENCY'],  refined_df['AVG_MONETARY_VALUE'])

  df['PRED_CLV']=ggf_actual.customer_lifetime_value(
  bgf_actual,
  df['FREQUENCY'],
  df['RECENCY'],
  df['AGE'],
  df['AVG_MONETARY_VALUE'],
  freq='W',
  time=time, 
  discount_rate=0.01
  )

  df['PROB_ALIVE'] =1
  df['PROB_ALIVE_t']=1

  return df[['CustomerID','FREQUENCY','RECENCY','AGE','AVG_MONETARY_VALUE','PRED_VISITS','PROB_ALIVE','PROB_ALIVE_t','PRED_CLV']]

# COMMAND ----------

t = 52.08 # 365 days
time = 12.0
sample = summaryDF.toPandas()

resultsDF = CLV_CustomerGroup.func(sample)
display(resultsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Takeaways
# MAGIC ![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) 
# MAGIC 
# MAGIC * Frequency is a distinct count for a given  time unit. Week or day
# MAGIC * We found weekly aggregation work better in terms of less error and better results
# MAGIC * We use 30 % of the data for validation, it is a consecutive period after the training. 6-12 month and predict next year.
# MAGIC * Cleansed the data by removing the one time business users
# MAGIC * The model did not perform well for Short-lived customers 
# MAGIC * a need to remove customers who are not part of the validation dataset
# MAGIC * We work with Finance department to calculate  profit/transaction as part of avg monetary value