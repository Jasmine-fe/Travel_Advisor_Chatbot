from typing import Literal
from operator import itemgetter
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class RouteMessage(TypedDict):
    """Route message to the appropriate service_type."""
    service_type: Literal[
        "tourist_attraction", 
        "itinerary_planning", 
        "restaurant_recommendations", 
        "exploring_travel_ideas",
        "others"
    ]
class ChatbotRouter:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.route_system = "Route the user's message to either 'tourist_attraction', 'itinerary_planning', 'restaurant_recommendations', 'exploring_travel_ideas', or 'others' if it doesn't fit into the previous categories."
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", self.route_system),
            ("human", "{message}"),
        ])
        self.route_chain = self.route_prompt | self.llm.with_structured_output(RouteMessage) | itemgetter("service_type")

    def route_message(self, message: str) -> str:
        return self.route_chain.invoke({"message": message})
