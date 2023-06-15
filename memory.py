from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0)

memory = (k=3)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)


conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
print(memory.buffer)
memory.load_memory_variables({})
