from travel_chatbot_chain import chatbot_chain
# main
if __name__ == "__main__":
    print("What can I help you with? (Type 'exit' to end)")
    user_input = input()
    
    while user_input.lower() != 'exit':
        input_data = {
            "message": user_input
        }
        response = chatbot_chain.invoke({'message': user_input})
        print(response)
        user_input = input()