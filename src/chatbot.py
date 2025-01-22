import os
import uuid
from dotenv import load_dotenv
from src.utils.long_term_memory import graph
from src.services.router_service import ChatbotRouter
from src.services.chatbot_service import ChatbotService
from src.chains.restaurant_chain import RestaurantChain

load_dotenv()

class Chatbot:
    def __init__(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        self.memory_config = {"configurable": {"user_id": str(uuid.uuid4()), "thread_id": "1"}}

        # Initialize regular services
        self.services = {
            "tourist_attraction": ChatbotService("You are an expert in finding tourist attractions."),
            "itinerary_planning": ChatbotService("You are an expert in travel itinerary planning."),
            "exploring_travel_ideas": ChatbotService("You are an expert in recommending travel ideas."),
            "others": ChatbotService("You are a friendly and helpful chatbot."),
            # Special case for restaurant recommendations
            "restaurant_recommendations": RestaurantChain().get_restaurant_chain()
        }

        self.router = ChatbotRouter()

    def process_message(self, message: str) -> str:
        # First, process the message through the memory graph
        memory_state = {"messages": [("user", message)]}
        memory_response = None
        
        for chunk in graph.stream(memory_state, config=self.memory_config):
            if "messages" in chunk.get("agent", {}):
                memory_response = chunk["agent"]["messages"][0].content
        
        # If memory agent provided a response, use it
        if memory_response:
            return memory_response
            
        # Otherwise, fall back to regular service routing
        service_type = self.router.route_message(message)
        print("service_type:", service_type)
        if service_type not in self.services:
            raise ValueError(f"Unknown service type: {service_type}")
            
        # Special handling for restaurant recommendations
        if service_type == 'restaurant_recommendations':
            return self.services[service_type].invoke(message)
        
        # Regular handling for other services
        return self.services[service_type].process_message(message)
