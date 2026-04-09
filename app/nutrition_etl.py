import argparse
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from modules import DataUtils,ModelUtils

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
    check_output=RunnableLambda(ModelUtils.check_nutrition_etl_output)

    nutrition_etl_chain=(prompt|model|check_output).with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(ValueError,)
    )

    try:
        results=nutrition_etl_chain.batch(image_paths)
    except ValueError as e:
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