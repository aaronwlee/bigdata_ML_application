from database import get_mongo_spark_for_thread

def sprak_write_file_to_mongo(file_name, file_path, set_event, clear_event):
    set_event()
    sc = get_mongo_spark_for_thread()
    """
    Load the csv file into spark and save it to MongoDB
    """
    df = sc.read.format('csv') \
        .option("inferSchema", True) \
        .option("header", True) \
        .option("sep", "\t") \
        .load(file_path)

    df.write.format("mongodb").mode("append").option("database", "bigdata").option("collection", file_name).save()
    clear_event()