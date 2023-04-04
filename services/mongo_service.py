from database import get_db
from werkzeug.local import LocalProxy
import math

# Use LocalProxy to read the global db instance with just `db`
db = LocalProxy(get_db)

def get_by_collection_with_page(collection, page):
    try:
        page_size = 15
        skips = page_size * (page - 1)
        cursor = db[collection].find().skip(skips).limit(page_size)
        return cursor
    except Exception as e:
        raise e
    
def insert_one_to_collection(collection, data):
    try:
        return db[collection].insert_one(data)
    except Exception as e:
        raise e

def get_collection_expain(collection):
    try:
        return list(db[collection].find_one().keys())
    except Exception as e:
        raise e
    
def get_collection_page_size(collection, limit=15):
    try:
        total = db[collection].count_documents({})
        print(total)
        size = int(math.ceil(total/limit))
        return size
    except Exception as e:
        raise e
    

def get_collection_list():
    try:
        return list(db.list_collection_names())
    except Exception as e:
        raise e
    
def drop_collection(collection):
    try:
        return db[collection].drop()
    except Exception as e:
        raise e