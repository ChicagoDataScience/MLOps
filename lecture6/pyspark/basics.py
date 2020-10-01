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

s3path = "s3://databricks-recsys/"
df.write.parquet(s3path+"u.parquet")

# COMMAND ----------

df_parquet = spark.read.parquet(s3path+"u.parquet").show()

# COMMAND ----------

pdf = df.toPandas()

# COMMAND ----------

pdf.head()

# COMMAND ----------

sdf = sqlContext.createDataFrame(pdf)

# COMMAND ----------

sdf.describe()

# COMMAND ----------

sdf.printSchema()

# COMMAND ----------

import databricks.koalas as ks
kdf = sdf.to_koalas()
kdf['iid'].to_numpy()[:3]

# COMMAND ----------

type(ks.from_pandas(pdf))

# COMMAND ----------

sdf.createOrReplaceTempView('sdf')

# COMMAND ----------

query = 'select distinct iid from sdf order by iid'
spark.sql(query).show()

# COMMAND ----------

movies_sdf = spark.read.csv("s3://databricks-recsys/movies_raw.dat",header=False, sep="|",inferSchema = True)
display(movies_sdf)

# COMMAND ----------

movies_sdf.createOrReplaceTempView('movies_sdf')

# COMMAND ----------

query = """
  select sdf.iid, avg(sdf.rating) as avg_rating, count(sdf.rating) as num_rating, first(movies_sdf._c1) as movie
  from sdf,movies_sdf
  where sdf.iid = movies_sdf._c0
  group by iid
  having num_rating >= 5
  order by avg_rating desc
  limit 10
"""
top_movies_sdf = spark.sql(query)

# COMMAND ----------

top_movies_kdf = top_movies_sdf.to_koalas()
top_movies_kdf.head()

# COMMAND ----------

display(top_movies_sdf)

# COMMAND ----------

sdf_grouped = sdf.groupBy("iid").agg({'rating':'avg'})
pdf_grouped = sdf_grouped.toPandas()
len(pdf_grouped)
