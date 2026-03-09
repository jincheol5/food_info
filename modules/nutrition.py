from typing import TypedDict,Any
from .prompts import build_nutrition_user_prompt
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,END

class State(TypedDict):
    vlm:Any
    parser:Any
    image_path:str
    vlm_output:str
    parserd_json:dict
    valid:bool

def vlm_node(state:State):
    print("Running VLM")
    vlm=state['vlm']
    prompt=HumanMessage(
        content=[
            {
                "type":"text",
                "text":build_nutrition_user_prompt()
            },
            {
                "type":"image_url",
                "image_url":{
                    "url":state['image_path']
                }
            }
        ]
    )
    response=vlm.invoke([prompt])
    state['vlm_output']=response.content
    return state

def parser_node(state:State):
    print("Parsing NutriFacts json with Pydantic")
    parser=state['parser']
    try:
        parsered_json=parser.parse(state["vlm_output"])
        state['parserd_json']=parsered_json
        state["valid"]=True
    except Exception as e:
        print("Parsing failed:",e)
        state["valid"]=False
    return state

def check_valid(state:State):
    if state["valid"]:
        print("Valid NutriFacts json")
        return END
    print("Invalid NutriFacts json, Retrying VLM extraction")
    return "vlm"

def get_nutrition_graph():
    graph=StateGraph(state_schema=State)
    graph.add_node(node="vlm",action=vlm_node)
    graph.add_node(node="parser",action=parser_node)
    graph.add_edge(start_key="vlm",end_key="parser")
    graph.set_entry_point(key="vlm") # graph 시작 노드 지정
    graph.add_conditional_edges(
        source="parser", # 조건 분기 수행할 노드
        path=check_valid, # 다음 노드를 결정하는 함수
        path_map={ # 조건 함수 반환값 -> 실제 다음 node 매핑
            "vlm":"vlm",
            END:END
        }
    )
    return graph.compile()