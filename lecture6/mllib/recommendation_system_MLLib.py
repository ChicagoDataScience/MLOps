# Databricks notebook source
# MAGIC %md In Cmd 2, the AWS_ACCESS_KEY and AWS_SECRET_KEY variables are set and kept hidden.

# COMMAND ----------

AWS_ACCESS_KEY = "AA"
AWS_SECRET_KEY = "BB"

# COMMAND ----------

sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_KEY)

# COMMAND ----------

df = spark.read.csv("s3://databricks-recsys/u.data",header=True, sep="\t",inferSchema = True)
display(df)

# COMMAND ----------

movies_sdf = spark.read.csv("s3://databricks-recsys/movies_raw.dat",header=False, sep="|",inferSchema = True)
display(movies_sdf)

# COMMAND ----------

ratings = df.rdd

numRatings = ratings.count()
numUsers = ratings.map(lambda r: r[0]).distinct().count()
numMovies = ratings.map(lambda r: r[1]).distinct().count()

print("Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrameNaFunctions as DFna
from pyspark.sql.functions import udf, col, when

movies_counts = df.groupBy(col("iid")).agg(F.count(col("rating")).alias("counts"))
movies_counts.show()

# COMMAND ----------

training_df, validation_df, test_df = df.randomSplit([.6, .2, .2], seed=594)

# COMMAND ----------

df.randomSplit?

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import Row
import numpy as np
import math

# COMMAND ----------

seed = 594
iterations = 10
regularization_parameter = 0.1
ranks = range(4, 12)
errors = []
err = 0
tolerance = 0.02

# COMMAND ----------

min_error = float('inf')
best_rank = -1
best_iteration = -1

for rank in ranks:
    als = ALS(maxIter=iterations, regParam=regularization_parameter, rank=rank, userCol="uid", itemCol="iid", ratingCol="rating")
    model = als.fit(training_df)
    predictions = model.transform(validation_df)
    new_predictions = predictions.filter(col('prediction') != np.nan)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(new_predictions)
    errors.append(rmse)

    print('For rank %s the RMSE is %s' % (rank, rmse))
    if rmse < min_error:
        min_error = rmse
        best_rank = rank
print('The best model was trained with rank %s' % best_rank)

# COMMAND ----------

training_df.take(3)

# COMMAND ----------

validation_df.take(3)

# COMMAND ----------

all_except_test_df = training_df.union(validation_df)

# COMMAND ----------

final_als = ALS(maxIter=10, regParam=0.1, rank=6, userCol="uid", itemCol="iid", ratingCol="rating")
final_model = final_als.fit(all_except_test_df)
final_pred = final_model.transform(test_df)
final_pred = final_pred.filter(col('prediction') != np.nan)
rmse = evaluator.evaluate(final_pred)
print("the one time final rmse (this is an internal metric) for our model is: {}".format(rmse))

# COMMAND ----------

np.random.seed(594)
user_id = np.random.choice(numUsers)

# COMMAND ----------

new_user_ratings = df.filter(df.uid == user_id)
new_user_ratings.sort('rating', ascending=True).take(10) # top rated movies for this user

# COMMAND ----------

new_user_ratings.describe('rating').show()

# COMMAND ----------

display(new_user_ratings)

# COMMAND ----------

new_user_rated_iids = [i.iid for i in new_user_ratings.select('iid').distinct().collect()]
movies_of_interest = [i.iid for i in movies_counts.filter(movies_counts.counts > 25).select('iid').distinct().collect()]
new_user_unrated_iids = list(set(movies_of_interest) - set(new_user_rated_iids))

# COMMAND ----------

import time
cols = ('uid', 'iid', 'timestamp')
new_user_preds = sqlContext.createDataFrame(zip([user_id] * len(new_user_unrated_iids), new_user_unrated_iids, [int(time.time())] * len(new_user_unrated_iids)), cols)
new_user_preds = final_model.transform(new_user_preds).filter(col('prediction') != np.nan)

# COMMAND ----------

new_user_preds.join(movies_sdf,new_user_preds.iid ==  movies_sdf._c0,"left").sort('prediction', ascending=False).take(10)

# COMMAND ----------

display(new_user_preds)

# COMMAND ----------


