# spark-apache.org MLlib collaborative filtering
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import explode, avg, count, col, round, row_number
#import pandas as pd
#import pyspark.sql.functions as SF
from pyspark.sql import Window

spark = SparkSession.builder.getOrCreate()

# df_yelp_business = spark.read.json("C:\\Users\\Adam\\Studies\\Documents\\mgr2\\big data\\restaurants\\yelp_academic_dataset_business.json", multiLine=True)
#df_yelp_business = spark.read.csv("C:\\Users\\Adam\\Studies\\Documents\\mgr2\\big data\\restaurants\\restaurants.csv",
#                                  sep=",", header=True, inferSchema=True)
df_yelp_business = spark.read.json("C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_business — kopia.json")\
    .withColumn("newbusinessid", row_number().over(Window.orderBy("business_id")))\
    .select('newbusinessid', 'business_id', 'name', 'city', 'address', 'categories')\
    .filter(df_yelp_business.categories.contains('Restaurants'))

df_yelp_business.show(10)
df_yelp_business.printSchema()


df_yelp_review = spark.read.json("C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_review.json")
df_yelp_review.show(10)
df_yelp_review.printSchema()

df_yelp_user = spark.read.json("C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_user.json")\
    .withColumn("newuserid", row_number().over(Window.orderBy("user_id")))

df_yelp_review = df_yelp_review.join(df_yelp_user, on="user_id", how='inner').select(df_yelp_review['business_id'], df_yelp_review['stars'], df_yelp_user['newuserid'])
df_yelp_review = df_yelp_review.join(df_yelp_restaurant, on = "business_id", how='inner').select(df_yelp_restaurant['newbusinessid'], df_yelp_review['stars'], df_yelp_review['newuserid'])



#qualifyRestaurants = df_yelp_review.groupBy("business_id").agg(avg("stars"), count("review_id")).filter(count("review_id") >= 100).withColumnRenamed("avg(stars)", "AverageRating").withColumnRenamed("count(review_id)", "CountRating")
#top10Restaurants = df_yelp_restaurant.join(qualifyRestaurants, on="business_id").select(df_yelp_business['name'], qualifyRestaurants["AverageRating"],  qualifyRestaurants["CountRating"]).orderBy(col("AverageRating").desc(),qualifyRestaurants["CountRating"].desc()).limit(100)
#top10Restaurants.show(100)

#podzial zbioru na dane trenujace i testowe
train, test = df_yelp_review.randomSplit([0.8, 0.2], 42)

#model rekomendacji dla użytkowników
als = ALS(maxIter=5, regParam=0.01, userCol="newuserid", itemCol="newbusinessid", ratingCol="stars", seed=42, coldStartStrategy="drop")
model = als.fit(train)
pred = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars", predictionCol="prediction")
a = evaluator.evaluate(pred)
userRecs = model.recommendForAllUsers(3)



#userRecsNames = userRecs.withColumn('recs', explode('recommendations'))
#userRecsNames = userRecsNames.select('newuserid', 'recs.*')
#userRecsNames = userRecsNames.join(df_yelp_restaurant, on='Book-ID', how='left').select(userRecsNames['*'],
#books['Book-Title'])
#userRecsNames.orderBy("User-ID").limit(10).show(truncate=False)




