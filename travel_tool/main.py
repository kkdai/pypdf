# Tools as OpenAI Functions
# Make sure langchain to 0.0.200
# pip install --upgrade --force-reinstall langchain

import json
from langchain.tools import MoveFileTool, format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from travel_tool.poi import TravelPOITool

model = ChatOpenAI(model="gpt-3.5-turbo-0613")

tools = [TravelPOITool()]
functions = [format_tool_to_openai_function(t) for t in tools]
print(functions[0])

while True:
    try:
        question = input("Question: ")
        hm = HumanMessage(content=question)
        ai_message = model.predict_messages([hm], functions=functions)
        ai_message.additional_kwargs['function_call']
        _args = json.loads(
            ai_message.additional_kwargs['function_call'].get('arguments'))
        tool_result = tools[0](_args)

        print("Answer: ", tool_result)
    except KeyboardInterrupt:
        break
