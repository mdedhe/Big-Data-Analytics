# Databricks notebook source
# DBTITLE 1,BigData of Telecom Churn
# MAGIC %md
# MAGIC Telecom customer churn prediction
# MAGIC 
# MAGIC This data set consists of 100 variables and approx 100 thousand records. This data set contains different variables explaining the attributes of telecom industry and various factors considered important while dealing with customers of telecom industry. The target variable here is churn which explains whether the customer will churn or not. We can use this data set to predict the customers who would churn or who wouldn't churn depending on various variables available.
# MAGIC 
# MAGIC 
# MAGIC <h3>References</h3><br>
# MAGIC Taking help for predicting the problem of churn using 
# MAGIC 
# MAGIC Spark MLlib documentation and the below mentioned link
# MAGIC :https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
# MAGIC 
# MAGIC 
# MAGIC Dataset Link https://www.kaggle.com/abhinav89/telecom-customer

# COMMAND ----------

# MAGIC %md
# MAGIC <br>Reading the data from the databricks cluster

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Telecom_customer_churn-edacf.csv"
file_type = "csv"

# CSV options
infer_schema = "True"
first_row_is_header = "True"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Changing the dataframe into the temperary table to perform the sql operations

# COMMAND ----------

# Create a view or table

temp_table_name = "Telecom_customer_churn_edacf_csv"
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC select * from `Telecom_customer_churn_edacf_csv`

# COMMAND ----------

# MAGIC %md
# MAGIC Changing the temparary table to permanent table for the future use 

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "Telecom_Customer_Churn"
#df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC select * from `Telecom_Customer_Churn`

# COMMAND ----------

#converting the SQL table into the Dataframe
Telecom_Data = sqlContext.sql("SELECT * FROM Telecom_Customer_Churn")

# COMMAND ----------

# DBTITLE 1,Data Preprocessing  
# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Data Preprocessing</h2>
# MAGIC   <p>
# MAGIC   1. Checking the datatypes before proceed for the data cleaning steps<br>
# MAGIC   2. Getting the total number of rows and the columns in the dataset<br>
# MAGIC   3. Checking for the base line curve using Histogram.
# MAGIC   4. Handling the Missing values from the dataset 
# MAGIC   5.
# MAGIC   </p>

# COMMAND ----------

#Confirming the Data Types for each variable in Data Frame
Telecom_Data.dtypes

# COMMAND ----------

print("Total Number of rows in the dataset are ", Telecom_Data.count(),"and columns are", len(Telecom_Data.columns))

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
Telecom_Data_Check = Telecom_Data.toPandas()
print(len(Telecom_Data_Check))
fig=plt.figure(figsize=(5,2))
sns.countplot(x='churn', data=Telecom_Data_Check, order=Telecom_Data_Check['churn'].value_counts().index)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC Above Distribution of Target variables shows there is no imbalance in target variables. Both the types are equally distributed, thus not going to implement Data Imbalance Handling. Now Looking into Missing Values.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Handling Missing Values

# COMMAND ----------

# we use the below function to find more information about the #missing values
import pandas as pd
def info_missing_table(df_pd):
    """Input pandas dataframe and Return columns with missing value and percentage"""
    mis_val = df_pd.isnull().sum() #count total of null in each columns in dataframe
#count percentage of null in each columns
    mis_val_percent = 100 * df_pd.isnull().sum() / len(df_pd) 
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) 
 #join to left (as column) between mis_val and mis_val_percent
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'}) 
#rename columns in table
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1) 
        
    print ("Your selected dataframe has " + str(df_pd.shape[1]) + " columns.\n"    #.shape[1] : just view total columns in dataframe  
    "There are " + str(mis_val_table_ren_columns.shape[0]) +              
    " columns that have missing values.") #.shape[0] : just view total rows in dataframe
    return mis_val_table_ren_columns
missings = info_missing_table(Telecom_Data_Check)
missings

# COMMAND ----------

# MAGIC %md
# MAGIC To handle missing values we tried one approach by replacing Numerical Values by Mean of Column and for Cateorical Values by Mode of Columns. But as it was going much deepeer into pandas it might fail into Big Data Architecture.
# MAGIC As per professors instruction in class, we are continuing with Spark and handing the missing values by simply dropping them.

# COMMAND ----------

#removing the all Null values from the columns 
Telecom_Data_New=Telecom_Data.dropna()

# COMMAND ----------

#printing the count after removing the null values from the dataset 
print("Total Number of rows in the new dataset are ", Telecom_Data_New.count(),"and columns are", len(Telecom_Data_New.columns))

# COMMAND ----------

#displaying the dataset
display(Telecom_Data_New)

# COMMAND ----------

Telecom_Data_New.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ####Converting Categorical Columns into Numerical

# COMMAND ----------

#converting the string variable into the numeric variable using string indexer
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="new_cell", outputCol="new_cellIndex")
indexer1 = StringIndexer(inputCol="crclscod", outputCol="crclscodIndex")
indexer2 = StringIndexer(inputCol="asl_flag", outputCol="asl_flagIndex")
indexer3 = StringIndexer(inputCol="prizm_social_one", outputCol="prizm_social_oneIndex")
indexer4 = StringIndexer(inputCol="area", outputCol="areaIndex")
indexer5 = StringIndexer(inputCol="dualband", outputCol="dualbandIndex")
indexer6 = StringIndexer(inputCol="refurb_new", outputCol="refurb_newIndex")
indexer7 = StringIndexer(inputCol="hnd_webcap", outputCol="hnd_webcapIndex")
indexer8 = StringIndexer(inputCol="ownrent", outputCol="ownrentIndex")
indexer9 = StringIndexer(inputCol="dwlltype", outputCol="dwlltypeIndex")
indexer10 = StringIndexer(inputCol="marital", outputCol="maritalIndex")
indexer11 = StringIndexer(inputCol="infobase", outputCol="infobaseIndex")
indexer12 = StringIndexer(inputCol="HHstatin", outputCol="HHstatinIndex")
indexer13 = StringIndexer(inputCol="dwllsize", outputCol="dwllsizeIndex")
indexer14 = StringIndexer(inputCol="ethnic", outputCol="ethnicIndex")
indexer15 = StringIndexer(inputCol="kid0_2", outputCol="kid0_2Index")
indexer16 = StringIndexer(inputCol="kid3_5", outputCol="kid3_5Index")
indexer17 = StringIndexer(inputCol="kid6_10", outputCol="kid6_10Index")
indexer18 = StringIndexer(inputCol="kid11_15", outputCol="kid11_15Index")
indexer19 = StringIndexer(inputCol="kid16_17", outputCol="kid16_17Index")
indexer20 = StringIndexer(inputCol="creditcd", outputCol="creditcdIndex")
#fitting the data into the indexers
indexed = indexer.fit(Telecom_Data_New).transform(Telecom_Data_New)
indexed1 = indexer1.fit(indexed).transform(indexed)
indexed2 = indexer2.fit(indexed1).transform(indexed1)
indexed3 = indexer3.fit(indexed2).transform(indexed2)
indexed4 = indexer4.fit(indexed3).transform(indexed3)
indexed5 = indexer5.fit(indexed4).transform(indexed4)
indexed6 = indexer6.fit(indexed5).transform(indexed5)
indexed7 = indexer7.fit(indexed6).transform(indexed6)
indexed8 = indexer8.fit(indexed7).transform(indexed7)
indexed9 = indexer9.fit(indexed8).transform(indexed8)
indexed10 = indexer10.fit(indexed9).transform(indexed9)
indexed11= indexer11.fit(indexed10).transform(indexed10)
indexed12= indexer12.fit(indexed11).transform(indexed11)
indexed13 = indexer13.fit(indexed12).transform(indexed12)
indexed14= indexer14.fit(indexed13).transform(indexed13)
indexed15 = indexer15.fit(indexed14).transform(indexed14)
indexed16 = indexer16.fit(indexed15).transform(indexed15)
indexed17 = indexer17.fit(indexed16).transform(indexed16)
indexed18 = indexer18.fit(indexed17).transform(indexed17)
Telecom_Data_New_Indexed = indexer20.fit(indexed18).transform(indexed18)

# COMMAND ----------

display(Telecom_Data_New_Indexed)

# COMMAND ----------

# MAGIC %md
# MAGIC Deleting all the columns with the string values because already the Integer value convertion is done.

# COMMAND ----------

listNew=["new_cell","crclscod","asl_flag","prizm_social_one","area","dualband","refurb_new","hnd_webcap","ownrent","dwlltype","marital","infobase","HHstatin","dwllsize","ethnic","kid0_2","kid3_5","kid6_10","kid11_15","kid16_17","creditcd"]

Telecom_Data_New_Indexed_Numeric = Telecom_Data_New_Indexed.drop(*listNew)

#Dispaying Final Cleaned Data for Machine Learning Models.
display(Telecom_Data_New_Indexed_Numeric)

# COMMAND ----------

# DBTITLE 1,Machine Learning Models
# MAGIC %md

# COMMAND ----------

#Importing Libraries 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import PCA
from pyspark.ml import Pipeline

# COMMAND ----------

assembler = VectorAssembler(
    inputCols=Telecom_Data_New_Indexed_Numeric.columns,
    outputCol="features")
output = assembler.transform(Telecom_Data_New_Indexed_Numeric)

#output.select("features", "clicked").show(truncate=False)

# COMMAND ----------

display(output)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Scaling the Data

# COMMAND ----------

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(output)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(output)

# COMMAND ----------

display(scaledData)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Principle Component Analysis

# COMMAND ----------

pca = PCA(k=30, inputCol="scaledFeatures", outputCol="pcaFeatures")
model = pca.fit(scaledData)
result = model.transform(scaledData).select("pcaFeatures","churn")

# COMMAND ----------

model.explainedVariance

# COMMAND ----------

# MAGIC %md
# MAGIC From above analysis of variance we can confirm only first few are adding significant variance into model. Thus there is no need for additional variabes. 

# COMMAND ----------

display(result)

# COMMAND ----------

data = result.selectExpr("pcaFeatures as features","churn as label")

# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Splitting the data into 70-30 split with the seed 42

# COMMAND ----------

train, test = data.randomSplit([0.7, 0.3], seed = 42)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Two-Class Logistic Regression

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)

# COMMAND ----------

predictions = lrModel.transform(test)
predictions.select('features', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

# COMMAND ----------

#Model Evaluation
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
ACC_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = ACC_evaluator.evaluate(predictions)
print("The accuracy of the model is {}".format(accuracy))

# COMMAND ----------

from sklearn.metrics import confusion_matrix
y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print("Below is the confusion matrix \n {}".format(cnf_matrix))

# COMMAND ----------

# DBTITLE 1,DECISION TREE
from pyspark.ml.classification import DecisionTreeClassifier

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
model = dt.fit(train)

# Make predictions.
predictions = model.transform(test)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g "% (accuracy))
print("Test Error = %g " % (1.0 - accuracy))

# COMMAND ----------

#Model Evaluation
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

#Confusion Matrix
y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print("Below is the confusion matrix \n {}".format(cnf_matrix))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Random Forest Classifier
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
model = rf.fit(train)

# Make predictions.
predictions = model.transform(test)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g "% (accuracy))
print("Test Error = %g " % (1.0 - accuracy))

# COMMAND ----------

#Model Evaluation
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

#Confusion Matrix
y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print("Below is the confusion matrix \n {}".format(cnf_matrix))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Gradient-boosted tree classifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

model = gbt.fit(train)

# Make predictions.
predictions = model.transform(test)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g "% (accuracy))
print("Test Error = %g " % (1.0 - accuracy))

# COMMAND ----------

#Model Evaluation
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

#Confusion Matrix
y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print("Below is the confusion matrix \n {}".format(cnf_matrix))

# COMMAND ----------

#Model Evaluation
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

#Confusion Matrix
y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print("Below is the confusion matrix \n {}".format(cnf_matrix))

# COMMAND ----------

# DBTITLE 1,SVC Support vector machine classification
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(train)

# Make predictions.
predictions = lsvcModel.transform(test)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = BinaryClassificationEvaluator()
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g "% (accuracy))
print("Test Error = %g " % (1.0 - accuracy))

# COMMAND ----------

#Model Evaluation
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

#Confusion Matrix
y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print("Below is the confusion matrix \n {}".format(cnf_matrix))

# COMMAND ----------

# MAGIC %md
# MAGIC Code by Group Number 6

# COMMAND ----------


