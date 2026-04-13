from pymongo import MongoClient
from pymongo.errors import PyMongoError

class DBInterface:
    def __init__(self,port:int=27017):
        try:
            self.client=MongoClient(f"mongodb://127.0.0.1:{port}/")
        except PyMongoError as e:
            print(f"MongoDB error: {e}")