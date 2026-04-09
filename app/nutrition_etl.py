import argparse
from pydantic import ValidationError
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from modules import DataUtils,ModelUtils,NutritionSchema

def nutrition_ETL(**kwargs):
    """
    """
    # 이미지 가져오기
    image_paths=DataUtils.get_nutrition_images()

    prompt=RunnableLambda(ModelUtils.get_nutrition_message)
    model=ChatOllama(
        model=app_config['model_name'],
        base_url=f"http://localhost:{kwargs['port']}"
    )
    model=model.with_structured_output(NutritionSchema) # 성공 시 결과값 NutritionSchema 객체, 실패 시 ValidationError
    # with_structured_output
    # 1. prompt에 형식 강제 내용 추가
    # 2. 출력 json 으로 파싱 
    # 3. pydantic 검증 .model_validate()

    nutrition_etl_chain=(prompt|model).with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(ValidationError,)
    )

    try:
        results=nutrition_etl_chain.batch(image_paths)
        results=[result.model_dump() for result in results] # NutritionSchema 객체들을 .model_dump 함수로 json 객체들로 변환
    except ValidationError as e:
        print("영양성분 정보 추출 실패: ",e)
        results=None
    if results is not None:
        print(results)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="qwen3.5:35b")
    parser.add_argument("--port",type=int,default=11434)
    args=parser.parse_args()
    app_config={
        # app 관련
        'model_name':args.model_name,
        'port':args.port
    }
    nutrition_ETL(**app_config)