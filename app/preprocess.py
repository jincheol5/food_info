from modules import DataUtils,DBInterface

### 식품 영양성분 이미지 목록 가져와서 MongoDB에 초기 저장
db_interface=DBInterface()
food_ids=DataUtils.get_food_list_from_imgs(img_type=f"nutrition")
db_interface.insert_food_ids(food_ids=food_ids)
print("식품 영양성분 정보 초기 입력 완료")