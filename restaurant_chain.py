import os
from dotenv import load_dotenv
from typing import Literal
from operator import itemgetter
from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore
from langchain_community.document_loaders.csv_loader import CSVLoader
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

class RestaurantType(TypedDict):
    restaurant_type: Literal["general", "michelin"]

class RestaurantChain:
    def __init__(self):
        """
        Initializes the RestaurantChain object with necessary components like LLM chain and prompt template.        
        """
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.retriever = None
        self.general_restaurant_chain  = None
        self.michelin_guide_chain = None
        self.restaurant_type_route_chain = None

    def build_michelin_guide_rag(self):
        ## Build a RAG for Michelin guide information
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
        # initialize a Chroma vector store and memory for storing and retrieving embedded documents
        vectorstore = Chroma(collection_name="full_documents", embedding_function=OpenAIEmbeddings())
        store = InMemoryStore()
        # text splitter for splitting child documents
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        # step4: ParentDocumentRetriever that combines the vector store, document store, and child document splitter
        self.retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
        self.retriever.add_documents(documents, ids=None)

    def build_restaurant_recommendation_chain(self):
        general_restaurant_template = """
            You are a restaurant recommendation chatbot, you need to provide relevant suggestions based on their preference.  
            Question: {question}
        """
        #TODO: optimize how to add additional information
        michelin_guide_template = """
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

        general_restaurant_prompt_template = ChatPromptTemplate.from_template(general_restaurant_template)
        michelin_guide_prompt_template = ChatPromptTemplate.from_template(michelin_guide_template)

        self.general_restaurant_chain = general_restaurant_prompt_template | self.llm | StrOutputParser()    
        self.michelin_guide_chain = {"context": self.retriever, "question": RunnablePassthrough()} | michelin_guide_prompt_template | self.llm | StrOutputParser()

    def define_restaurant_type_route_chain(self):
        route_system = "Route the user's query to either 'general' or 'michelin'"
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", route_system),
                ("human", "{query}"),
            ]
        )
        # Routing chain based on the query
        self.restaurant_type_route_chain = route_prompt | self.llm.with_structured_output(RestaurantType) | itemgetter("restaurant_type")

    def get_restaurant_recommendation_result(self, restaurant_type, query):
        if restaurant_type == 'general':
            print("Get restaurant_type = general")
            res = self.general_restaurant_chain.invoke(query)
        elif restaurant_type == 'michelin':
            print("Get restaurant_type = michelin")
            res = self.michelin_guide_chain.invoke(query)
        else:
            res = None
        return res

    def get_full_chains(self):
        self.build_michelin_guide_rag()
        self.build_restaurant_recommendation_chain()
        self.define_restaurant_type_route_chain()

        restaurant_chain = {
            "restaurant_type": self.restaurant_type_route_chain,
            "query": lambda x: x['query']
        } | RunnableLambda(
            lambda x: self.get_restaurant_recommendation_result(restaurant_type=x["restaurant_type"], query=x['query'])
        )
        return restaurant_chain