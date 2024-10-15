import os
import streamlit as st
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
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']

class RestaurantType(TypedDict):
    restaurant_type: Literal["general", "michelin"]

class RestaurantChain:
    def __init__(self):
        """
        Initializes the RestaurantChain object with necessary components like LLM chain and prompt template.        
        """
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        
        self.michelin_guide_chain = None
        self.embeddings = OpenAIEmbeddings()
        self.retriever = self.build_michelin_guide_rag()
        self.general_restaurant_chain = self.build_general_recommendation_chain()
        self.michelin_guide_chain = self.build_michelin_recommendation_chain()
        self.restaurant_type_route_chain = self.define_restaurant_type_route_chain()


    def build_michelin_guide_rag(self):
        ## Build a RAG for Michelin guide information
        vectorstore = Chroma(collection_name="michelin_guide_restaurants", embedding_function=self.embeddings)
        
        # Check if data already exists in Chroma
        if vectorstore.get()['ids']:
            print("Loading existing data from Chroma...")
        else:
            print("No existing data found. Loading from CSV and storing in Chroma...")
            self._load_and_store_data(vectorstore)

        store = InMemoryStore()
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
        return retriever

    def _load_and_store_data(self, vectorstore):
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
        vectorstore.add_documents(documents)

    def build_general_recommendation_chain(self):
        general_restaurant_template = """
            You are a restaurant recommendation chatbot, you need to provide relevant suggestions based on their preference.  
            Question: {question}
        """
        general_restaurant_prompt_template = ChatPromptTemplate.from_template(general_restaurant_template)
        general_restaurant_chain = general_restaurant_prompt_template | self.llm | StrOutputParser()    
        return general_restaurant_chain

    def build_michelin_recommendation_chain(self):
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

        michelin_guide_prompt_template = ChatPromptTemplate.from_template(michelin_guide_template)
        michelin_guide_chain = {"context": self.retriever, "question": RunnablePassthrough()} | michelin_guide_prompt_template | self.llm | StrOutputParser()
        return michelin_guide_chain

    def define_restaurant_type_route_chain(self):
        route_system = "Route the user's query to either 'general' or 'michelin'"
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", route_system),
                ("human", "{message}"),
            ]
        )
        restaurant_type_route_chain = route_prompt | self.llm.with_structured_output(RestaurantType) | itemgetter("restaurant_type")
        return restaurant_type_route_chain

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

    def get_restaurant_chain(self):
        restaurant_chain = {
            "restaurant_type": self.restaurant_type_route_chain,
            "message": lambda x: x['message']
        } | RunnableLambda(
            lambda x: self.get_restaurant_recommendation_result(restaurant_type=x["restaurant_type"], query=x['message'])
        )
        return restaurant_chain