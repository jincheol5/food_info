from typing import TypedDict,Any
from .prompts import build_ingredient_user_prompt
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,END

class State(TypedDict):
    vlm:Any
    parser:Any
    image_path:str
    vlm_output:str
    parserd_json:dict
    valid:bool
    failed_examples:list[str]

def vlm_node(state:State):
    print("Running VLM")
    vlm=state['vlm']

    fail_examples_text=""
    if state.get("fail_examples"):
        fail_examples_text=f"\n\nPrevious invalid outputs (DO NOT repeat):\n"
        for i,ex in enumerate(state["fail_examples"]):
            fail_examples_text+=f"\nInvalid Example {i+1}:\n{ex}\n"
    prompt=HumanMessage(
        content=[
            {
                "type":"text",
                "text":build_ingredient_user_prompt()+fail_examples_text
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
    print("Parsing Ingredients json with Pydantic")
    parser=state['parser']
    try:
        parsered_json=parser.parse(state["vlm_output"])
        state['parserd_json']=parsered_json
        state["valid"]=True
    except Exception as e:
        print("Parsing failed:",e)
        state["valid"]=False
        if "fail_examples" not in state:
            state["fail_examples"]=[]
        state["fail_examples"].append(state["vlm_output"])
    return state

def check_valid(state:State):
    if state["valid"]:
        print("Valid Ingredients json")
        return END
    print("Invalid Ingredients json, Retrying VLM extraction")
    return "vlm"

def get_ingredient_graph():
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