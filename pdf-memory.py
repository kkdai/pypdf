from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.chains import ChatVectorDBChain  # for chatting with the pdf
from langchain.chains import ConversationChain
from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})


# llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferWindowMemory(k=3)
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
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
print(memory.buffer)
