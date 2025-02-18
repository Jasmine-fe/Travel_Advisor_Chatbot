# LLM Travel Chatbot with Michelin Restaurant RAG

## Overview

This project is a sophisticated travel chatbot powered by a large language model (LLM) that provides users with personalized travel-related services. The chatbot leverages OpenAI's API to interact with users, offering recommendations and insights tailored to their travel preferences. A key feature of this chatbot is its ability to recommend Michelin-starred restaurants through a Retrieval-Augmented Generation (RAG) approach.

https://github.com/user-attachments/assets/094c5d54-4e59-45f1-99dc-c86333721bb3


## Features

The chatbot offers a variety of services, including:

- **Tourist Attraction Recommendations**: Discover popular tourist spots based on user preferences or location.
- **Itinerary Planning**: Assist users in planning their travel itineraries by suggesting optimal routes, activities, and schedules.
- **Restaurant Recommendations**: Provide personalized restaurant suggestions, including Michelin-starred options, based on user preferences.
- **Exploring Travel Ideas**: Inspire users with new travel destinations or experiences tailored to their interests.
- **Miscellaneous Travel Assistance**: Offer additional travel-related information and support not covered by the main services.

## Project Structure

The project consists of the following key components:

- `app.py`: The main application file that runs the chatbot interface using Streamlit.
- `chatbot.py`: Contains the core logic for processing user messages and routing them to the appropriate services.
- `chains/restaurant_chain.py`: Implements the restaurant recommendation logic, including the RAG approach for Michelin restaurants.
- `services/chatbot_service.py`: Defines the chatbot service that interacts with the LLM.
- `services/router_service.py`: Routes user messages to the appropriate service based on their content.
- `utils/build_RAG_db.py`: Builds and persists the Michelin restaurant database for retrieval.
- `utils/long_term_memory.py`: Manages long-term memory for the chatbot to enhance user interactions.

## Usage

To run the chatbot, execute the following command:

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Acknowledgments

- OpenAI for providing the API and models that power the chatbot.
- The Langchain library for facilitating the implementation of RAG and LLM interactions.
