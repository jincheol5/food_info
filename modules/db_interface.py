from datetime import datetime,timezone
from pymongo import MongoClient
from pymongo.errors import PyMongoError

class DBInterface:
    def __init__(self,port:int=27017):
        try:
            self.client=MongoClient(f"mongodb://127.0.0.1:{port}/")
            self.db=self.client["chronolab"]
        except PyMongoError as e:
            print(f"MongoDB error: {e}")
    
    def connect_db(self,port:int=27017):
        try:
            self.client=MongoClient(f"mongodb://127.0.0.1:{port}/")
            self.db=self.client["chronolab"]
        except PyMongoError as e:
            print(f"MongoDB error: {e}")
    
    def disconnect_db(self):
        self.client.close()
    
    def insert_food_ids(self,food_ids:list):
        if self.client is None:
            self.connect_db()
        food_info_collection=self.db["food_info"]
        
        food_info_map={
            food_id:{
                "_id":food_id,
                "food_id":food_id,
                "extracted":False,
                "validated":False,
                "created_t":datetime.now(timezone.utc), # BSON의 Date 타입 (ISODate)으로 저장
                "extracted_t":None, # update 시 datetime.now(timezone.utc)
                "validated_t":None, # update 시 datetime.now(timezone.utc)
                "nutrition":{},
                "ingredient":{}
            }
            for food_id in food_ids
        }
        food_info_list=list(food_info_map.values())
        try:
            food_info_collection.insert_many(food_info_list,ordered=False) # ordered=False: 중복 _id 있으면 skip
        except PyMongoError as e:
            print(f"MongoDB insert error: {e}")

    def get_unextracted_food_ids(self):
        if self.client is None:
            self.connect_db()
        food_info_collection=self.db["food_info"]
        food_ids=list(
            doc["food_id"]
            for doc in food_info_collection.find(
                {"extracted":False,"validated":False}, # 조건 (filter)
                {"_id":1} # 가져올 필드 (projection)
            )
        )
        return food_ids

    def update_batch_nutrition_info(self,nutrition_info_list:list):
        if self.client is None:
            self.connect_db()
        food_info_collection=self.db["food_info"]
        
        for nutrition_info in nutrition_info_list:
            """
            nutrition_info
                (food_id,result)
            """
            food_id=nutrition_info[0]
            result=nutrition_info[1]
            try:
                food_info_collection.update_one(
                    {"_id":food_id},
                    {
                        "$set":{
                            "extracted":True,
                            "extracted_t":datetime.now(timezone.utc),
                            "nutrition":result
                        }
                    }
                )
            except PyMongoError as e:
                print(f"MongoDB update error: {e}")
                continue
