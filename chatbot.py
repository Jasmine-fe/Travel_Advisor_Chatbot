import os
import streamlit as st
from typing import Dict
from dotenv import load_dotenv
from chatbot_router import ChatbotRouter
from chatbot_service import ChatbotService
from restaurant_chain import RestaurantChain

load_dotenv()

class Chatbot:
    def __init__(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        self.service_prompts = {
            "tourist_attraction": "You are an expert in finding tourist attractions.",
            "itinerary_planning": "You are an expert in travel itinerary planning.",
            "exploring_travel_ideas": "You are an expert in recommending travel ideas.",
            "restaurant_recommendations": "You are an expert in restaurant recommendations.",
            "others": "You are a friendly and helpful chatbot."
        }

        self.services: Dict[str, ChatbotService] = {
            service: ChatbotService(prompt) for service, prompt in self.service_prompts.items()
        }

        # Special case for restaurant recommendations
        restaurant_chain = RestaurantChain().get_restaurant_chain()
        self.services['restaurant_recommendations'] = restaurant_chain

        self.router = ChatbotRouter()

    def process_message(self, message: str) -> str:
        service_type = self.router.route_message(message)
        print("service_type:", service_type)
        if service_type not in self.services:
            raise ValueError(f"Unknown service type: {service_type}")
            
        # Special handling for restaurant recommendations
        if service_type == 'restaurant_recommendations':
            return self.services[service_type].invoke(message)
        
        # Regular handling for other services
        return self.services[service_type].process_message(message)