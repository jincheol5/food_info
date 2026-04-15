import os
from typing import Literal

class DataUtils:
    """
    """
    dir_path=os.path.join("raw_images")
    food_info_path=os.path.join("images")
    
    @staticmethod
    def get_food_list_from_raw_imgs():
        food_list_path=os.path.join(DataUtils.dir_path)
        food_list=[
            name for name in os.listdir(food_list_path)
            if os.path.isdir(os.path.join(food_list_path,name))
        ]
        return food_list
    
    @staticmethod
    def get_food_list_from_imgs(img_type:Literal["nutrition","ingredient"]):
        food_list_path=os.path.join(DataUtils.food_info_path,img_type)
        food_list=[
            os.path.splitext(name)[0] 
            for name in os.listdir(food_list_path)
            if os.path.isfile(os.path.join(food_list_path,name))
        ]
        return food_list

    @staticmethod
    def get_food_images(food_id):
        food_imgs_path=os.path.join(DataUtils.dir_path,f"{food_id}")
        image_paths=[
            os.path.join(food_imgs_path,name) 
            for name in os.listdir(food_imgs_path)
            if os.path.isfile(os.path.join(food_imgs_path,name))
        ]
        return image_paths
    
    @staticmethod
    def get_nutrition_images():
        food_imgs_path=os.path.join(DataUtils.food_info_path,"nutrition")
        image_paths=[
            os.path.join(food_imgs_path,name) 
            for name in os.listdir(food_imgs_path)
            if os.path.isfile(os.path.join(food_imgs_path,name))
        ]
        return image_paths

    @staticmethod
    def get_img_paths_of_food_ids(food_ids:list,img_type:Literal["nutrition","ingredients"]="nutrition"):
        image_paths=[
            os.path.join(DataUtils.food_info_path,"nutrition",f"{food_id}.png")
            for food_id in food_ids
        ]
        return image_paths
    