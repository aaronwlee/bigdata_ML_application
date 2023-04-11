import os

from flask import current_app, g
from flask_pymongo import PyMongo

from bson.objectid import ObjectId
from pyspark.sql import SparkSession

from flask.json import JSONEncoder
from bson import json_util, ObjectId
from datetime import datetime

from werkzeug.local import LocalProxy

class MongoJsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, ObjectId):
            return str(obj)
        return json_util.default(obj, json_util.CANONICAL_JSON_OPTIONS)
    

def get_db():
    """
    Configuration method to return db instance
    """
    db = getattr(g, "_database", None)

    if db is None:

        db = g._database = PyMongo(current_app).db
       
    return db

def get_mongo_spark():
    sc = getattr(g, "_spark", None)

    if sc is None:

        sc = g._spark = SparkSession \
            .builder \
            .appName("myApp") \
            .config("spark.mongodb.read.connection.uri", os.getenv("MONGO_URI")) \
            .config("spark.mongodb.write.connection.uri", os.getenv("MONGO_URI")) \
            .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:10.1.1') \
            .getOrCreate()
       
    return sc

def get_mongo_spark_for_thread():
    sc = SparkSession \
            .builder \
            .appName("myApp") \
            .config("spark.mongodb.read.connection.uri", os.getenv("MONGO_URI")) \
            .config("spark.mongodb.write.connection.uri", os.getenv("MONGO_URI")) \
            .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:10.1.1') \
            .getOrCreate()
    
    return sc


sc = LocalProxy(get_mongo_spark)
