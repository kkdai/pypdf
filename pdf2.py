from langchain.document_loaders import PyPDFLoader  # for loading the pdf
# for creating embeddings&#8203;``oaicite:{"number":2,"metadata":{"title":"langchain.embeddings.openai — 🦜🔗 LangChain 0.0.195","url":"https://python.langchain.com/en/latest/_modules/langchain/embeddings/openai.html","text":"class OpenAIEmbeddings(BaseModel, Embeddings):\n        \"\"\"Wrapper around OpenAI embedding models.\n\n        To use, you should have the ``openai`` python package installed, and the\n        environment variable ``OPENAI_API_KEY`` set with your API key or pass it\n        as a named parameter to the constructor.\n\n        Example:\n            .. code-block:: python\n\n                from langchain.embeddings import OpenAIEmbeddings\n                openai = OpenAIEmbeddings(openai_api_key=\"my-api-key\")\n\n        In order to use the library with Microsoft Azure endpoints, you need to set\n        the OPENAI_API_TYPE, OPENAI_API_BASE, OPENAI_API_KEY and OPENAI_API_VERSION.\n        The OPENAI_API_TYPE must be set to 'azure' and the others correspond to\n        the properties of your endpoint.\n        In addition, the deployment name must be passed as the model parameter.\n\n        Example:\n            .. code-block:: python\n\n                import os\n                os.environ[\"OPENAI_API_TYPE\"] = \"azure\"\n                os.environ[\"OPENAI_API_BASE\"] = \"https://<your-endpoint.openai.azure.com/\"\n                os.environ[\"OPENAI_API_KEY\"] = \"your AzureOpenAI key\"\n                os.environ[\"OPENAI_API_VERSION\"] = \"2023-03-15-preview\"\n                os.environ[\"OPENAI_PROXY\"] = \"http://your-corporate-proxy:8080\"\n\n                from langchain.embeddings.openai import OpenAIEmbeddings\n                embeddings = OpenAIEmbeddings(\n                    deployment=\"your-embeddings-deployment-name\",\n                    model=\"your-embeddings-model-name\",\n                    openai_api_base=\"https://your-endpoint.openai.azure.com/\",\n                    openai_api_type=\"azure\",\n                )\n                text = \"This is a test query.\"\n                query_result = embeddings.embed_query(text)\n\n        \"\"\"\n\n        client: Any  #: :meta private:\n        model: str = \"text-embedding-ada-002\"\n        deployment: str = model  # to support Azure OpenAI Service custom deployment names\n        openai_api_version: Optional[str] = None\n        # to support Azure OpenAI Service custom endpoints\n        openai_api_base: Optional[str] = None\n        # to support Azure OpenAI Service custom endpoints\n        openai_api_type: Optional[str] = None\n        # to support explicit proxy for OpenAI\n        openai_proxy: Optional[str] = None\n        embedding_ctx_length: int = 8191\n        openai_api_key: Optional[str] = None\n        openai_organization: Optional[str] = None\n        allowed_special: Union[Literal[\"all\"], Set[str]] = set()\n        disallowed_special: Union[Literal[\"all\"], Set[str], Sequence[str]] = \"all\"\n        chunk_size: int = 1000\n        \"\"\"Maximum number of texts to embed in each batch\"\"\"\n        max_retries: int = 6\n        \"\"\"Maximum number of retries to make when generating.\"\"\"\n        request_timeout: Optional[Union[float, Tuple[float, float]]] = None\n        \"\"\"Timeout in seconds for the OpenAPI request","pub_date":null}}``&#8203;
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma  # for the vectorization part&#8203;``oaicite:{"number":3,"metadata":{"title":"Get all documents from ChromaDb using Python and langchain - Stack Overflow","url":"https://stackoverflow.com/questions/76184540/get-all-documents-from-chromadb-using-python-and-langchain","text":"I'm using langchain to process a whole bunch of documents which are in an Mongo database.\n\nI can load all documents fine into the chromadb vector storage using langchain. Nothing fancy being done here. This is my code:\n\n    from langchain.embeddings.openai import OpenAIEmbeddings\n    embeddings = OpenAIEmbeddings()\n\n    from langchain.vectorstores import Chroma\n    db = Chroma.from_documents(docs, embeddings, persist_directory='db')\n    db.persist()\n\nNow, after storing the data, I want to get a list of all the documents and embeddings WITH id's.\n\nThis is so I can store them back into MongoDb.\n\nI also want to put them through Bertopic to get the topic categories.\n\nQuestion 1 is: how do I get all documents I've just stored in the Chroma database? I want the documents, and all the metadata","pub_date":null}}``&#8203;
# for chatting with the pdf&#8203;``oaicite:{"number":4,"metadata":{"title":"langchain.chains.conversational_retrieval.base — 🦜🔗 LangChain 0.0.195","url":"https://python.langchain.com/en/latest/_modules/langchain/chains/conversational_retrieval/base.html","text":"ChatVectorDBChain` is deprecated - \"\n                \"please use `from langchain.chains import ConversationalRetrievalChain","pub_date":null}}``&#8203;
from langchain.chains import ConversationalRetrievalChain
# I couldn't find the new class for OpenAI, so I'm assuming it's the same

pdf_path = "./paper.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(pages[0].page_content)

# 2. Creating embeddings and Vectorization
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embeddings=embeddings,
                                 persist_directory=".")
vectordb.persist()

# 3. Querying
# Here I assume that the method to create an instance of the ConversationalRetrievalChain class is similar to the deprecated ChatVectorDBChain class
# But I couldn't find the updated documentation for this class, so this part might need to be adjusted
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                               vectordb, return_source_documents=True)

query = "What is the bitcoin?"
result = pdf_qa({"question": query, "chat_history": ""})
print("Answer:")
print(result["answer"])
