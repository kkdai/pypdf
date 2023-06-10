import click
from dotenv import load_dotenv
from langchain.chains import ChatVectorDBChain  # for chatting with the pdf
from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
from langchain.vectorstores import Chroma  # for the vectorization part
from loguru import logger


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
    # print(pages[0].page_content)

    # 2. Creating embeddings and Vectorization
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(pages,
                                     embedding=embeddings,
                                     persist_directory=".")
    vectordb.persist()

    # 3. Querying
    llm = OpenAI(temperature=0.9, model_name=model_name)
    pdf_qa = ChatVectorDBChain.from_llm(llm,
                                        vectordb,
                                        return_source_documents=True)

    query = "What is the bitcoin?"
    result = pdf_qa({"question": query, "chat_history": ""})
    logger.info('Answer: {}', result["answer"])
