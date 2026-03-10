from modules import IngredientSchema,build_igredient_user_prompt
from langchain_core.output_parsers import PydanticOutputParser

# parser=PydanticOutputParser(pydantic_object=IngredientSchema)
# format_instructions=parser.get_format_instructions()
print(build_igredient_user_prompt())
