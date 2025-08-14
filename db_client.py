# MongoDB helper for Conversational Banking
from pymongo import MongoClient
import os

def get_db():
    import streamlit as st
    uri = st.secrets.get("MONGO_URI", "")
    if not uri:
        return None
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    db_name = st.secrets.get("MONGO_DATABASE", "conversational_banking")
    return client[db_name]

def mongo_ping():
    import streamlit as st
    uri = st.secrets.get("MONGO_URI", "")
    if not uri:
        return False, "MONGO_URI not set"
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.server_info()
        return True, "MongoDB connection successful"
    except Exception as e:
        return False, str(e)
