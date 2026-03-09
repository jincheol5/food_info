import argparse
import requests

def main(app_config:dict):
    match app_config['app_num']:
        case 1:
            """
            model pull
            """
            response=requests.post(
                f"http://localhost:{app_config['port']}/api/pull",
                json={"name":app_config['model_name']}
            )
            print(response.status_code)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--port",type=int,default=11434) # default port=11434
    parser.add_argument("--model_name",type=str,default="qwen2.5:1.5b") # Qwen2.5-1.5B-Instruct
    args=parser.parse_args()
    app_config={
        # app 관련
        'app_num':args.app_num,
        'port':args.port,
        'model_name':args.model_name
    }
    main(app_config=app_config)