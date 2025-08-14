# db_client.py
import certifi
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import streamlit as st

def get_mongo_client() -> MongoClient:
    """
    Builds a TLS-verified Mongo client using Streamlit secrets
    and a modern CA bundle (certifi). Works in Streamlit Cloud.
    """
    uri = st.secrets["MONGO_URI"]            # mongodb+srv://... (password URL-encoded)
    max_pool = int(st.secrets.get("MONGODB_MAX_POOL_SIZE", 10))
    min_pool = int(st.secrets.get("MONGODB_MIN_POOL_SIZE", 0))

    client = MongoClient(
        uri,
        tls=True,
        tlsCAFile=certifi.where(),           # CRITICAL for TLS on Streamlit Cloud
        maxPoolSize=max_pool,
        minPoolSize=min_pool,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=20000,
    )
    return client

def get_db():
    client = get_mongo_client()
    db_name = st.secrets["MONGO_DATABASE"]
    return client[db_name]

def mongo_ping() -> tuple[bool, str]:
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        return True, "MongoDB ping OK"
    except ServerSelectionTimeoutError as e:
        return False, f"Server selection timeout: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
