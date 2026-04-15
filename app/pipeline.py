from modules import DataUtils,DBInterface,FoodInfoChain
from pydantic import ValidationError

### 식품 영양성분 이미지 목록 가져와서 MongoDB에 초기 저장
db_interface=DBInterface()
# food_ids=DataUtils.get_food_list_from_imgs(img_type=f"nutrition")
# db_interface.insert_food_ids(food_ids=food_ids)
# print("식품 영양성분 정보 초기 입력 완료")

### 식품 영양성분 정보 추출 후 MongoDB 업데이트
food_ids=db_interface.get_unextracted_food_ids()
image_paths=DataUtils.get_img_paths_of_food_ids(food_ids=food_ids,img_type=f"nutrition")

def chunk_list(lst,batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

food_ids_batch_list=chunk_list(lst=food_ids,batch_size=2)
image_paths_batch_list=chunk_list(lst=image_paths,batch_size=2)

nutrition_extract_chain=FoodInfoChain.get_nutrition_extract_chain(
    model_name=f"qwen3.5:35b",
    port=11434
)

for idx,(food_ids,image_paths) in enumerate(zip(food_ids_batch_list,image_paths_batch_list)):
    print(f"processing {idx+1} batch...")
    try:
        results=nutrition_extract_chain.batch(image_paths)
        results=[(food_id,result.model_dump()) for food_id,result in zip(food_ids,results)] # NutritionSchema 객체들을 .model_dump 함수로 json 객체들로 변환
        db_interface.update_batch_nutrition_info(nutrition_info_list=results)
        print(f"{idx+1} batch 영양성분 정보 추출 성공")
    except ValidationError as e:
        print(f"{idx+1} batch 영양성분 정보 추출 실패: ",e)
        results=None
print(f"batch processing end")