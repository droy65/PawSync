"""MongoDB connection (PyMongo) + indexes. Falls back to mongomock when MONGO_URI is empty."""
import os
from pymongo import MongoClient, ASCENDING, DESCENDING
_db = None
def get_db(fresh=False):
    global _db
    if _db is not None and not fresh:
        return _db
    uri = os.getenv("MONGO_URI")
    if uri:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    else:
        import mongomock
        client = mongomock.MongoClient()
    _db = client["pet_adoption_db"]
    _db.users.create_index("email", unique=True)
    _db.pets.create_index([("status", ASCENDING), ("species", ASCENDING), ("age_months", ASCENDING)])
    _db.pets.create_index([("shelter_id", ASCENDING)])
    _db.pets.create_index([("location.city", ASCENDING)])
    _db.applications.create_index([("adopter_id", ASCENDING), ("pet_id", ASCENDING)])
    _db.applications.create_index([("shelter_id", ASCENDING), ("status", ASCENDING)])
    _db.interactions.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
    _db.favorites.create_index([("user_id", ASCENDING), ("pet_id", ASCENDING)], unique=True)
    return _db
