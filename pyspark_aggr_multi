# Databricks notebook source
# MAGIC %md
# MAGIC ##More PySpark techniques

# COMMAND ----------

import pyspark.sql.functions as psf
from pyspark.sql import Window, Row
from datetime import date

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1) Partitioning vs Grouping

# COMMAND ----------

# MAGIC %md
# MAGIC Partitioning and Grouping are techniques to apply aggregate functions on subset of data rather than on the entire dataframe. There is one big difference however.
# MAGIC  1. **Grouping** will reduce the number of rows in your DataFrame.
# MAGIC  2. **Partitioning** just divides your data in subsets so you can run the aggregating function later on. This is very useful technique, not available in Pandas. Cherry on the pie: you do not lose data provenance when aggregating data.
# MAGIC
# MAGIC Let's see a concrete example for each.

# COMMAND ----------

# Let's create a PySpark dataframe from scratch, one row at a time
df = spark.createDataFrame([
    Row(SUBJID='1', VISIT=1, HR=80,  AGE=25, SEX='M'),
    Row(SUBJID='1', VISIT=2, HR=85,  AGE=25, SEX='M'),
    Row(SUBJID='2', VISIT=1, HR=90,  AGE=44, SEX='F'),
    Row(SUBJID='2', VISIT=2, HR=75,  AGE=44, SEX='F'),
    Row(SUBJID='3', VISIT=1, HR=105, AGE=21, SEX='M'),
    Row(SUBJID='3', VISIT=2, HR=95,  AGE=21, SEX='M'),
    Row(SUBJID='4', VISIT=1, HR=70,  AGE=36, SEX='F'),
    Row(SUBJID='4', VISIT=2, HR=65,  AGE=36, SEX='F'),
])

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### a) Grouping

# COMMAND ----------

# Get the average Heart Rate for each subject:
df.groupBy('SUBJID').mean('HR').display()

# Note that the number of rows has been reduced. This is a summary.

# COMMAND ----------

# Similarly, the youngest by gender
df.groupBy('SEX').min('AGE').display()

# Here again, the number of rows has been reduced

# COMMAND ----------

# MAGIC %md
# MAGIC ##### b) Partitioning

# COMMAND ----------

# Let's partition the dataframe by SUBJID. Each partition can be processed in parallel across different nodes in a cluster, which can improve performance for large datasets.

window = Window.partitionBy('SUBJID') # Window creates the partition. Why 'Window'? Thinks about the view you get from a window: a subset of what's out there.

# We can use that window to create a new column using an aggregate function, like F.min
df2 = df.withColumn(
    "min_HR", 
    psf.min("HR").over(window)
)

# We have ADDED to our df the minimal HR per SUBJID. The number of rows remained unchanged. This is nice, no need to merge anything and we kept all the columns
df2.display()

# COMMAND ----------

# We can use the same technique, and the same window object btw, to calculate 'HR Baseline' and 'Change from baseline': 

df3 = df.withColumn(
        "Baseline",
        psf.first(psf.when(psf.col("VISIT") == 1, psf.col("HR")), ignorenulls=True).over(window)    
    )\
    .withColumn(
        "Chg_from_baseline",
        psf.when(
            psf.col('Visit') != 1, psf.col("HR") - psf.col("Baseline")
        )
    )\
    .orderBy('SUBJID', 'VISIT')

df3.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2) collect() and collect_set()

# COMMAND ----------

# MAGIC %md
# MAGIC Use collect() when you need to retrieve all data (including duplicates) from a DataFrame to the driver node (bringing all the data locally).
# MAGIC
# MAGIC Use collect_set() when you want to aggregate unique values within groups in the DataFrame.

# COMMAND ----------

# Here's how we can get all SEX values, and put them in a list. Less elegant than Pandas syntax...
[row["SEX"] for row in df.select("SEX").collect()]

# COMMAND ----------

# collect_set() gets all the UNIQUE values for a given column, and put them inside an array
df.agg(psf.collect_set("SEX")).display()

# For each SUBJID, get all unique HR values in a string, with a comma as delimiter
df.groupBy("SUBJID").agg(psf.array_join(psf.collect_set("HR"), ', ').alias("All_HR")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3) Data provenance when aggregating data 

# COMMAND ----------

# When we aggregate data, we can lose the data provenance. Without it the listing template, specifically addDataProvenance(), will fail.
# Say we want to add to DM domain the average HR by subject, coming from VS. We want to push this later on to the silver layer.

basic_DM = spark.createDataFrame([
    Row(SUBJID='1', AGE=25, SEX='M', DM_URECID='1657912664161'),
    Row(SUBJID='2', AGE=44, SEX='F', DM_URECID='9873131643217'),
])

basic_VS = spark.createDataFrame([
    Row(SUBJID='1', VISIT=1, HR=80,  VS_URECID='103241324165'),
    Row(SUBJID='1', VISIT=2, HR=85,  VS_URECID='237566541315'),
    Row(SUBJID='2', VISIT=1, HR=120, VS_URECID='787465415615'),
    Row(SUBJID='2', VISIT=2, HR=75,  VS_URECID='023413354787'),
])

# COMMAND ----------

# Aggregating the data using groupBy/agg will not keep the data provenance. We do NOT recommend using it in that scenario
VS_summary = basic_VS.groupBy("SUBJID").agg(psf.mean("HR").alias("Mean_HR"))

# See, at this point, data provenance (VS_URECID in my example) was lost:
VS_summary.display()

# COMMAND ----------

# Aggregating the data using partitionBy is the best way here, since we don't lose the data provenance. 

window = Window.partitionBy('SUBJID')
basic_VS = basic_VS.withColumn(
    'HR_mean',
    psf.mean("HR").over(window)
)

basic_VS.display()

# COMMAND ----------

# If we need to keep only one record, we can always use dropduplicates(), but remember to sort before to avoid non-deterministic issues
basic_VS.orderBy('VS_URECID').dropDuplicates(['SUBJID']).display()

# We can merge this onto DM by SUBJID, and we'll have provenance from both DM and VS.

# COMMAND ----------

# In some other scenarios (e.g. pivot/unpivot), it won't be possible to use partitionBy. In that case, we need to bring back arbritrary provenance. 
# Let's filter VS to keep only one record for each SUBJID.

# This is one way. Create a rank column, and filter on it.
VS_unique  = basic_VS.withColumn('rank', psf.rank().over(Window.partitionBy('SUBJID').orderBy('VS_URECID'))) # Deterministic sort, otherwise the result will not be deterministic and saving to silver layer will fail!
VS_unique = VS_unique.filter(psf.col('rank') == 1).drop('rank')

# We have provenance info
VS_unique.display()

# COMMAND ----------

# Merge the summary against this nodup VS:
df = VS_unique.join(VS_summary, "SUBJID", "left")

# We have now a summary with data provenance, though arbritrary. And we can merge this onto DM, and get data provenance from both domains.
df.display()

# COMMAND ----------
