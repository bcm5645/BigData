# Projekt na zaliczenie przedmiotu BigData
# PŁ, Infomratyka 2 stopnia, niestacjonarne
# Adam Jędrzejec 244012, Mariusz Leśnik 244022, Wiktor Marczak 244026
  
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import explode, avg, count, col, round, row_number
from pyspark.sql import Window

from datetime import datetime
from pathlib import Path
import os.path

spark = SparkSession.builder.config("spark.driver.memory", "32g").getOrCreate()
print(str(datetime.now()) + ": Wczytanie danych o firmach")
# ścieżka do spreparowanego zestawu danych o firmach
path_to_file = "C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_restaurant.json"
path = Path()
if not os.path.exists(path_to_file): # jeśli nie istnieje spreparowany zestaw to zaczytaj oryginalny i spreparuj
    df_yelp_business = spark.read.json("C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_business.json")\
        .select('business_id', 'name', 'city', 'address', 'categories')

    print(str(datetime.now()) + ": Filtrowanie restauracji")
    df_yelp_business = df_yelp_business\
        .filter(df_yelp_business.categories.contains('Restaurants'))\
        .withColumn("businessid", row_number().over(Window.orderBy("business_id")))
    df_yelp_business.write.json(path_to_file)

df_yelp_business = spark.read.json(path_to_file)    # wczytanie danych o firmach
df_yelp_business.printSchema()
df_yelp_business.show(10)

print(str(datetime.now()) + ": Wczytanie danych o ocenach")
# ścieżka do spreparowanego zestawu danych o ocenach
path_to_file = "C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_review1.json"
path = Path()
if not os.path.exists(path_to_file): # jeśli nie istnieje spreparowany zestaw to zaczytaj oryginalny i spreparuj
    df_yelp_review = spark.read.json("C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_review.json")
    df_yelp_review = df_yelp_review.select(['business_id','stars','user_id'])
    df_yelp_review.write.json(path_to_file)

df_yelp_review = spark.read.json(path_to_file) # wczytanie danych o ocenach
df_yelp_review.printSchema()
df_yelp_review.show(10)

df_yelp_review.printSchema()

print(str(datetime.now()) + ": Wczytanie danych o użytkownikach")
# ścieżka do spreparowanego zestawu danych o użytkownikach
path_to_file = "C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_user1.json"
path = Path()
if not os.path.exists(path_to_file): # jeśli nie istnieje spreparowany zestaw to zaczytaj oryginalny i spreparuj
    df_yelp_user = spark.read.json("C:\\Users\\v460g\OneDrive - Politechnika Łódzka\\Big Data\Projekt\\Yelp\\yelp_academic_dataset_user.json")\
    .select(['user_id', 'name'])\
    .withColumn("newuserid", row_number().over(Window.orderBy("user_id")))
    df_yelp_user.write.json(path_to_file)

df_yelp_user = spark.read.json(path_to_file)
df_yelp_user.show(10)

print(str(datetime.now()) + ": Filtrowanie ocen tylko do restauracji")
df_yelp_review = df_yelp_review\
    .join(df_yelp_business, on = "business_id", how='inner')\
    .select(df_yelp_business['businessid'], df_yelp_review['stars'], df_yelp_review['user_id'], df_yelp_business['name'])

df_yelp_review = df_yelp_review\
    .join(df_yelp_user, on="user_id", how='inner')\
    .select(df_yelp_review['businessid'], df_yelp_review['stars'], df_yelp_user['newuserid'], df_yelp_review['name'])
df_yelp_user.printSchema()

qualifyRestaurants = df_yelp_review\
    .groupBy("businessid")\
    .agg(avg("stars"), count("stars"))\
    .filter(count("stars") >= 500)\
    .withColumnRenamed("avg(stars)", "AverageRating")\
    .withColumnRenamed("count(stars)", "CountRating")
top50Restaurants = df_yelp_business\
    .join(qualifyRestaurants, on="businessid")\
    .select(df_yelp_business['name'], qualifyRestaurants["AverageRating"],  qualifyRestaurants["CountRating"])\
    .orderBy(col("AverageRating").desc(),qualifyRestaurants["CountRating"].desc())\
    .limit(50)
top50Restaurants.show()

print(str(datetime.now()) + ": Przygotowanie zbiorów testowego i treningowego")
#podział na dane treningowe i testowe
train, test = df_yelp_review.randomSplit([0.8, 0.2], 42)
train.show(10)

print(str(datetime.now()) + ": Trenowanie modelu rekomendacji")
#model rekomendacji
als = ALS(maxIter=5, regParam=0.01, userCol="newuserid", itemCol="businessid", ratingCol="stars", seed=42, coldStartStrategy="drop")
model = als.fit(train)

print(str(datetime.now()) + ": Przetwarzanie modelu rekomendacji")
pred = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars", predictionCol="prediction")
print(str(datetime.now()) + ": Ewaluacja modelu")
a = evaluator.evaluate(pred)
print(str(datetime.now()) + ": Rekomendacja")
userRecs = model.recommendForAllUsers(3)
userRecs.show(50)
# print(str(datetime.now()) + ": Koniec opracowania modelu")
# userRecsNames = userRecs.withColumn('recs', explode('recommendations'))
# userRecsNames = userRecsNames.select('newuserid', 'recs.*')
# userRecsNames = userRecsNames.join(df_yelp_business , on='businessid', how='left').select(userRecsNames['*'], df_yelp_business['name'])
# userRecsNames.orderBy("newuserid").limit(10).show(truncate=False)
# print(str(datetime.now()) + ": Koniec")

