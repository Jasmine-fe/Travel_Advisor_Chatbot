import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore
from langchain.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# ## Build a RAG for Michelin guide information

# step1: read source data and provide data description 
# read source data and provide data description 
michelin_guide_restaurants_path = "dataset/test_canada_michelin_guide_restaurants_Aug2024.csv"
loader = CSVLoader(
    file_path=michelin_guide_restaurants_path,
    source_column='Name',
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ['Name', 'Address', 'Location', 'Price', 'Cuisine', 'Longitude', 'Latitude', 'PhoneNumber', 'Url', 'WebsiteUrl', 'Award', 'GreenStar', 'FacilitiesAndServices', 'Description']
    },
)
documents = loader.load()


# step2: initialize a Chroma vector store and memory for storing and retrieving embedded documents
vectorstore = Chroma(collection_name="full_documents", embedding_function=OpenAIEmbeddings())
store = InMemoryStore()


# step3: text splitter for splitting child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)


# step4: ParentDocumentRetriever that combines the vector store, document store, and child document splitter
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)
retriever.add_documents(documents, ids=None)


# step5: create a prompt template for the chatbot
template = """
Information to understand the context: 
Data: michelin guide restaurants 
Columns: 
    Name: The name of the restaurant.
    Address: Full address, including city, region, and country.
    Location: City and country of the restaurant.
    Price: The average cost range represented by symbols (e.g., $, $$, $$$).
    Cuisine: Type of cuisine offered by the restaurant.
    Coordinates: Longitude and latitude for mapping the restaurant’s location.
    Contact Information: Phone number and URLs for the Michelin Guide page and the restaurant’s official website.
    Award: Michelin distinction (e.g., Michelin Star, Bib Gourmand, Selected Restaurants).
    GreenStar: Indicates whether the restaurant has a Michelin Green Star for sustainability.
    Description: A brief summary highlighting the restaurant's ambiance, menu, and standout dishes.

Answer the question based on the following context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# step6: create a processing chain that combines context retrieval, prompt formatting, and model response parsing
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()


# step7: invoke the processing chain using the input query to retrieve relevant information and generate a response
query = "please list Japanese cuisine restaurants with 2 star award"
res = chain.invoke(query)

