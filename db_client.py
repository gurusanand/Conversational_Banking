# MongoDB helper for Conversational Banking
from pymongo import MongoClient
import os

def get_db():
    uri = os.getenv("MONGO_URI", "")
    if not uri:
        return None
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    db_name = os.getenv("MONGO_DB", "conversational_banking")
    return client[db_name]

def mongo_ping():
    uri = os.getenv("MONGO_URI", "")
    if not uri:
        return False, "MONGO_URI not set"
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.server_info()
        return True, "MongoDB connection successful"
    except Exception as e:
        return False, str(e)
