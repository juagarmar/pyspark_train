# Databricks notebook source
# MAGIC %md
# MAGIC # Let's practice PySpark !

# COMMAND ----------

# MAGIC %md
# MAGIC **Please do not push your modifications to the Bitbucket repo, at least not in the main branch.**

# COMMAND ----------

# Let's create some practice data

from pyspark.sql import Row
from datetime import date

DM = spark.createDataFrame([
    Row(SUBJID='1', AGE=25, SEX='F',  HEIGHT=150,  SCREENED=date(2024, 1, 15)  ),
    Row(SUBJID='2', AGE=56, SEX='M',  HEIGHT=181,  SCREENED=date(2024, 2, 10)  ),
    Row(SUBJID='3', AGE=44, SEX='?',  HEIGHT=170,  SCREENED=date(2023, 8, 17)  ),
    Row(SUBJID='4', AGE=27, SEX='F',  HEIGHT=None, SCREENED=date(2024, 2, 16)  ),
    Row(SUBJID='5', AGE=46, SEX=None, HEIGHT=173,  SCREENED=None               ),
    Row(SUBJID='6', AGE=51, SEX='F',  HEIGHT=164,  SCREENED=date(2023, 11, 10) ),    
])

VS = spark.createDataFrame([
    Row(SUBJID=1, VISIT='SCREENING', VSTESTCD='SYSBP', VALUE=120   ),
    Row(SUBJID=1, VISIT='SCREENING', VSTESTCD='DIABP', VALUE=70    ),
    Row(SUBJID=1, VISIT='SCREENING', VSTESTCD='HR',    VALUE=86    ),
    Row(SUBJID=1, VISIT='DAY 1'   ,  VSTESTCD='SYSBP', VALUE=131   ),
    Row(SUBJID=1, VISIT='DAY 1'   ,  VSTESTCD='DIABP', VALUE=72    ),
    Row(SUBJID=1, VISIT='DAY 1'   ,  VSTESTCD='HR',    VALUE=None  ),    
    Row(SUBJID=3, VISIT='SCREENING', VSTESTCD='SYSBP', VALUE=111   ),
    Row(SUBJID=3, VISIT='SCREENING', VSTESTCD='DIABP', VALUE=75    ),
    Row(SUBJID=3, VISIT='SCREENING', VSTESTCD='HR',    VALUE=92    ),
    Row(SUBJID=4, VISIT='DAY 1'   ,  VSTESTCD='SYSBP', VALUE=118   ),
    Row(SUBJID=4, VISIT='DAY 1'   ,  VSTESTCD='DIABP', VALUE=80    ),
    Row(SUBJID=4, VISIT='DAY 1'   ,  VSTESTCD='HR',    VALUE=91    ),
    Row(SUBJID=6, VISIT='SCREENING', VSTESTCD='SYSBP', VALUE=110   ),
    Row(SUBJID=6, VISIT='SCREENING', VSTESTCD='DIABP', VALUE=65    ),
    Row(SUBJID=6, VISIT='SCREENING', VSTESTCD='HR',    VALUE=75    ),
    Row(SUBJID=6, VISIT='DAY 1'   ,  VSTESTCD='SYSBP', VALUE=95    ),
    Row(SUBJID=6, VISIT='DAY 1'   ,  VSTESTCD='DIABP', VALUE=152   ),
    Row(SUBJID=6, VISIT='DAY 1'   ,  VSTESTCD='HR',    VALUE=97    ),      
])

DM.display()
VS.display()

# COMMAND ----------

# Exercice #1
# In DM:
#   - Add a STUDYID colum in DM and VS, with value 'TEST_STUDY'
#   - Create a USUBJID columnn in DM and VS, concatenating STUDYID and SUBJID, with an hyphen between them
#   - Replace missing screening dates by '2023-01-01'

# COMMAND ----------

# Exercice #2
# Create a column DM.SEXL. Value should be:
#   'Female'  when SEX='F'
#   'Male'    when SEX='M'
#   'Unknown' for other non-missing SEX values
#   'Missing' for missing SEX values

# COMMAND ----------

# Exercice #3
# Create a summary table counting the number of subjects in each SEXL category

# COMMAND ----------

# Exercice #4
# How many male subjects are 40 years old or less?

# COMMAND ----------

# Exercice #5
# In DM:
#   - Create a new column 'IDEAL_WEIGHT'. It's a string like 'WEIGHT1-WEIGHT2' where WEIGHT1 is the weight for which BMI would be 20 and WEIGHT2 is the weight for which BMI would be 25. Both weights should be rounded integers.
#   - If you dare:
#       - add a WEIGHT column with a random value from 50 to 100. 
#       - Create a flag with values 'UNDERWEIGHT'|'NORMAL'|'OVERWEIGHT'|'MISSING' based on the BMI (thresholds are 20 and 25).

# COMMAND ----------

# Exercice #6
#   - How many subjects do have at least one VS record?
#   - How many subjects do have a VS record for all scheduled visits (i.e. 'SCREENING' and 'DAY1')?

# COMMAND ----------

# Exercice #7
# In DM: 
#   - flag subjects with a HR at screening higher than 90. Create a new column called 'HIGH_SCREENING_HR' which can be True, False or Null
#   - print to the log a (python) list of subjects where HIGH_SCREENING_HR is True

# COMMAND ----------

# Exercice #8
# In VS:
#   - Create a Baseline column which takes the value for each test at VISIT='Screening'
#   - Create a Change_from_Baseline column

# COMMAND ----------

# Exercice #9
# What is the average screening HR value, by SEX?

# COMMAND ----------

# Exercice #10
# Filter DM to display subjects screened in 2024, and sort it by descending screening date.

# COMMAND ----------

# Exercice #11
# Create a listing to highlight subjects with a diastolic blood pressure higher than their systolic blood pressure, in the same visit. Data Management should take a look at it ^-^

# COMMAND ----------

# Exercice #12
# Some subjects don't have VS records. Be creative and generate fake records to be added to the VS DataFrame.

# COMMAND ----------

# Exercice #13
# Be creative and produce a listing of your own creation using DM and VS dataFrames.
