import argparse
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from modules import DataUtils,ModelUtils

def img_classifier(**kwargs):
    """
    """
    # 이미지 가져오기
    food_list=DataUtils.get_food_list()
    image_paths=DataUtils.get_food_images(
        food_id=food_list[kwargs["food_num"]]
    )

    prompt=RunnableLambda(ModelUtils.get_classifier_message)
    model=ChatOllama(
        model=app_config['model_name'],
        base_url=f"http://localhost:{kwargs['port']}"
    )
    check_output=RunnableLambda(ModelUtils.check_classifier_output)

    img_classifier_chain=(prompt|model|check_output).with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(ValueError,)
    )

    try:
        results=img_classifier_chain.batch(image_paths)
    except ValueError as e:
        print("식품 이미지 분류 실패: ",e)
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
    img_classifier(**app_config)