"""Run a linear regression using Apache Spark ML.

In the following PySpark (Spark Python API) code, we take the following actions:

  * Load a previously created linear regression (BigQuery) input table
    into our Cloud Dataproc Spark cluster as an RDD (Resilient
    Distributed Dataset)
  * Transform the RDD into a Spark Dataframe
  * Vectorize the features on which the model will be trained
  * Compute a linear regression using Spark ML

"""

from datetime import datetime
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import NaiveBayes,LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier


# import matplotlib.pyplot as plt

# The imports, above, allow us to access SparkML features specific to linear
# regression as well as the Vectors types.

# Define a function that collects the features of interest
# (mother_age, father_age, and gestation_weeks) into a vector.
# Package the vector in a tuple containing the label (`weight_pounds`) for that
# row.

def vector_from_inputs(r):
  return (r["rain"], Vectors.dense(int(r["station_number"]),
                                            int(r["month"]),
                                            float(r["mean_temp"]),
                                            float(r["mean_dew_point"]),
                                            float(r["mean_visibility"]),
                                            float(r["mean_wind_speed"]),
                                            float(r["max_sustained_wind_speed"]),
                                            float(r["max_temperature"]),
                                            float(r["total_precipitation"])))

# Use Cloud Dataprocs automatically propagated configurations to get
# the Cloud Storage bucket and Google Cloud Platform project for this
# cluster.
sc = SparkContext()
spark = SparkSession(sc)
bucket = spark._jsc.hadoopConfiguration().get("fs.gs.system.bucket")
project = spark._jsc.hadoopConfiguration().get("fs.gs.project.id")

# Set an input directory for reading data from Bigquery.
todays_date = datetime.strftime(datetime.today(), "%Y-%m-%d-%H-%M-%S")
input_directory = "gs://{}/tmp/natality-{}".format(bucket, todays_date)

# Set the configuration for importing data from BigQuery.
# Specifically, make sure to set the project ID and bucket for Cloud Dataproc,
# and the project ID, dataset, and table names for BigQuery.

conf = {
    # Input Parameters
    "mapred.bq.project.id": project,
    "mapred.bq.gcs.bucket": bucket,
    "mapred.bq.temp.gcs.path": input_directory,
    "mapred.bq.input.project.id": project,
    "mapred.bq.input.dataset.id": "spark",
    "mapred.bq.input.table.id": "rain_ten",
}

# Read the data from BigQuery into Spark as an RDD.
table_data = spark.sparkContext.newAPIHadoopRDD(
    "com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat",
    "org.apache.hadoop.io.LongWritable",
    "com.google.gson.JsonObject",
    conf=conf)

# Extract the JSON strings from the RDD.
table_json = table_data.map(lambda x: x[1])

# Load the JSON strings as a Spark Dataframe.
rain_data = spark.read.json(table_json)
# Create a view so that Spark SQL queries can be run against the data.
rain_data.createOrReplaceTempView("rain")

# As a precaution, run a query in Spark SQL to ensure no NULL values exist.
sql_query = """
SELECT *
from rain
WHERE mean_temp is not null and
mean_dew_point is not null and
mean_visibility is not null and
mean_wind_speed is not null and 
max_sustained_wind_speed is not null and 
max_temperature is not null and 
total_precipitation is not null and 
rain is not null
"""
clean_data = spark.sql(sql_query)
# print(type(clean_data))
clean_data = clean_data.withColumn('rain', regexp_replace('rain', 'false', '0'))
clean_data = clean_data.withColumn('rain', regexp_replace('rain', 'true', '1'))
clean_data = clean_data.withColumn("rain", clean_data["rain"].cast(IntegerType()))


# #Summary statistics for numeric variables
# numeric_features = [t[0] for t in clean_data.dtypes if t[1] == 'int']
# double_features  = [t[0] for t in clean_data.dtypes if t[1] == 'double']
# for i in double_features:
#     numeric_features.append(i)
# numeric_features.remove('_c0')
# clean_data.select(numeric_features).describe().toPandas().transpose()

# #Correlations between independent variables
# numeric_data = clean_data.select(numeric_features).toPandas()
# axs = scatter_matrix(numeric_data, alpha=0.2, figsize=(6, 6), diagonal='kde')
# n = len(numeric_data.columns)
# for i in range(n):
#     v = axs[i, 0]
#     v.yaxis.label.set_rotation(0)
#     v.yaxis.label.set_ha('right')
#     v.set_yticks(())
#     h = axs[n-1, i]
#     h.xaxis.label.set_rotation(90)
#     h.set_xticks(())




# Create an input DataFrame for Spark ML using the above function.
training_data = clean_data.rdd.map(vector_from_inputs).toDF(["label",
                                                             "features"])
print(type(training_data))


training_data.cache()
train,test = training_data.randomSplit([0.7, 0.3], seed = 2018)

# Construct a new SVM object and fit the training data.
svm = LinearSVC(maxIter=5, regParam=0.01)
model = svm.fit(train)
# Print the model summary.
print("Coefficients:" + str(model.coefficients))
print ("Intercept:" + str(model.intercept))
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))


# nb = LogisticRegression(regParam=0.01,maxIter=1000,featuresCol = 'features', labelCol = 'label')
# model_lr = nb.fit(train)
# print "Coefficients:" + str(model_lr.coefficients)
# print "Intercept:" + str(model_lr.intercept)
# predictions = model_lr.transform(test)
# evaluator = BinaryClassificationEvaluator()
# print('Test Area Under ROC', evaluator.evaluate(predictions))

# # #coefficients plot 
# beta = np.sort(model_lr.coefficients)
# plt.plot(beta)
# plt.ylabel('Beta Coefficients')
# plt.show()

# # #ROC curve 
# trainingSummary = model_lr.summary
# roc = trainingSummary.roc.toPandas()
# plt.plot(roc['FPR'],roc['TPR'])
# plt.ylabel('False Positive Rate')
# plt.xlabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()
# print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# # #precision and recall
# pr = trainingSummary.pr.toPandas()
# plt.plot(pr['recall'],pr['precision'])
# plt.ylabel('Precision')
# plt.xlabel('Recall')
# plt.show()

# #decision tree
# dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 10)
# dtModel = dt.fit(train)
# predictions = dtModel.transform(test)
# evaluator = BinaryClassificationEvaluator()
# print('Test Area Under ROC', evaluator.evaluate(predictions))

