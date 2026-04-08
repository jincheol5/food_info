import argparse


def img_classifier(**kwargs):
    """
    """


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