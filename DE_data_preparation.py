# Databricks notebook source
# File location and type
file_location = "/FileStore/tables/online_retail_train.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS customer_info;
# MAGIC DROP TABLE IF EXISTS customer_info.online_retail_train

# COMMAND ----------

permanent_table_name = "customer_info.online_retail_train"
df.write.format("delta").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM customer_info.online_retail_train

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT SUM(profit_value) AS profit_value,country 
# MAGIC FROM customer_info.online_retail_train
# MAGIC GROUP BY country
# MAGIC ORDER BY profit_value DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT Description,StockCode FROM customer_info.online_retail_train

# COMMAND ----------

# MAGIC %md Summary Table

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Summary_2011.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

permanent_table_name = "customer_info.Summary_2011"

df.write.format("delta").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM customer_info.Summary_2011

# COMMAND ----------

