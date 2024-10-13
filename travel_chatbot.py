import os
import json
import pandas as pd
from typing import Literal
from dotenv import load_dotenv
from operator import itemgetter
from langchain_chroma import Chroma
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from restaurant_chain import RestaurantChain
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def chain_creator(llm, prompt_template, memory):
    return (
        RunnablePassthrough.assign(
            chat_history=lambda x: memory.chat_memory.messages
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

def create_prompt_template(system_message: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{message}")
        ]
    )

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define system messages for each service
service_prompts = {
    "tourist_attraction": "You are an expert in finding tourist attractions.",
    "itinerary_planning": "You are an expert in travel itinerary planning.",
    "exploring_travel_ideas": "You are an expert in recommending travel ideas.",
    "others": "You are a friendly and helpful chatbot."
}

# Create memory for each service
service_memories = {service: ConversationBufferMemory(return_messages=True) for service in service_prompts.keys()}

# Create chains for each service (service without Rag)
service_chains = {
    service: chain_creator(
        llm=llm, 
        prompt_template=create_prompt_template(service_prompts[service]),
        memory=service_memories[service]
    ) for service in service_prompts.keys()
}
# Create chains for the restaurant recommendation service
restaurantChain = RestaurantChain()
restaurant_chain = restaurantChain.get_restaurant_chain()
service_chains['restaurant_recommendations'] = restaurant_chain


# Define the routing system
class RouteMessage(TypedDict):
    """Route message to the appropriate service_type."""
    service_type: Literal[
        "tourist_attraction", 
        "itinerary_planning", 
        "restaurant_recommendations", 
        "exploring_travel_ideas",
        "others"
    ]

route_system = "Route the user's message to either 'tourist_attraction', 'itinerary_planning', 'restaurant_recommendations', 'exploring_travel_ideas', or 'others' if it doesn't fit into the previous categories."

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        ("human", "{message}"),
    ]
)

# Routing chain based on the message
route_chain = route_prompt | llm.with_structured_output(RouteMessage) | itemgetter("service_type")

# Final chain that routes to the appropriate LLM chain based on user input
def get_chain_for_service(input):
    service_type = input['service_type']
    message = input['message']
    print("service_type: ", service_type)
    print("message: ", message)
    if service_type in service_chains.keys():
        response = service_chains.get(service_type).invoke({"message": message})
        # Add the interaction to memory
        service_memories[service_type].chat_memory.add_user_message(message)
        service_memories[service_type].chat_memory.add_ai_message(response)
        return response
    else:
        raise ValueError(f"Unknown service type: {service_type}")
    
# Combine all the steps into the final chain
chatbot_chain = {"service_type": route_chain, "message": lambda x: x['message']} | RunnableLambda(get_chain_for_service)