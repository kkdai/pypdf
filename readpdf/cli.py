import click
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma


@click.command()
@click.argument('pdf-path', type=click.Path(exists=True))
@click.option('-m',
              '--model-name',
              type=click.STRING,
              default='gpt-3.5-turbo',
              help='model name')
@click.option('--dotenv-path',
              type=click.Path(),
              default='.env',
              help='path to .env file')
def main(pdf_path, model_name, dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # Creating embeddings and Vectorization
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(pages,
                                     embedding=embeddings,
                                     persist_directory=".")
    vectordb.persist()

    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)

    # Querying
    llm = ChatOpenAI(temperature=0.9, model_name=model_name)
    chain = ConversationalRetrievalChain.from_llm(llm,
                                                  vectordb.as_retriever(),
                                                  memory=memory)

    while True:
        try:
            question = input("Question: ")
            result = chain({"question": question})
            print("Answer: ", result["answer"])
        except KeyboardInterrupt:
            break
