# Travel Chatbot Using OpenAI API

This project is a travel chatbot built using OpenAI's API that provides helpful travel-related services. The chatbot uses the latest language model from OpenAI to interact with users and offer recommendations or insights related to travel. Whether you're exploring new destinations, looking for a place to eat, or planning your next trip, this chatbot is designed to help.

## Features

The chatbot offers five key services:
- **Tourist Attraction**: Find recommendations for tourist spots based on the user's preferences or location.
- **Itinerary Planning**: Help users plan their trip itineraries by suggesting the best routes, activities, and schedules.
- **Restaurant Recommendations**: Recommend restaurants based on user preferences, including specific requests like finding Michelin-starred restaurants.
- **Exploring Travel Ideas**: Inspire users by suggesting new travel destinations or experiences based on their interests.
- **Others**: Provide miscellaneous travel-related information and assistance not covered by the main services.

## Project Structure

- `main.py`: The main script to run the chatbot, which initializes and integrates all services using OpenAI’s API.
- `restaurant_chain.py`: Contains functionality related to recommending restaurants, including an internal data set of Michelin-starred restaurants.
- `travel_chatbot.py`: Core logic of the chatbot, defining service types and how the chatbot handles different user requests.