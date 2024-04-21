# Databricks notebook source
import polars as pl
import seaborn as sns

# COMMAND ----------

df = pl.read_csv("/Volumes/catalog1/schema1/data/iris.csv")

# COMMAND ----------

sns.barplot(data=df, x="species", y="petal_length")

# COMMAND ----------

df.columns

# COMMAND ----------

p = sns.displot(data=df, 
                x="sepal_width", hue="species", col="species", 
                height=3, aspect=1, alpha=1)

# COMMAND ----------


