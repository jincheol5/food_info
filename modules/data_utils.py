import os

class DataUtils:
    """
    """
    dir_path=os.path.join("raw_images")
    
    @staticmethod
    def get_food_list():
        food_list_path=os.path.join(DataUtils.dir_path)
        food_list=[
            name for name in os.listdir(food_list_path)
            if os.path.isdir(os.path.join(food_list_path,name))
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