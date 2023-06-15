# Tools as OpenAI Functions
# Make sure langchain to 0.0.200
# pip install --upgrade --force-reinstall langchain

from langchain.tools import MoveFileTool, DeleteFileTool, format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

model = ChatOpenAI(model="gpt-3.5-turbo-0613")


tools = [MoveFileTool()]
functions = [format_tool_to_openai_function(t) for t in tools]

message = model.predict_messages(
    [HumanMessage(content='move file foo to bar')], functions=functions)

print(message)

print(message.additional_kwargs['function_call'])

del_tools = [DeleteFileTool()]
del_functions = [format_tool_to_openai_function(t) for t in del_tools]

message = model.predict_messages(
    [HumanMessage(content='delete *.md from Document folder')], functions=del_functions)

print(message)

print(message.additional_kwargs['function_call'])
