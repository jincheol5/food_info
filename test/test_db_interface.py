import argparse

def test_db_interface(**kwargs):
    """
    """
    match kwargs["test_num"]:
        case 1:
            """
            """


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    parser.add_argument("--port",type=int,default=11434)
    args=parser.parse_args()
    test_config={
        # app 관련
        'test_num':args.test_num,
        'port':args.port
    }
    test_db_interface(**test_config)