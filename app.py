import streamlit as st
from src.chatbot import Chatbot

chatbot = Chatbot()

# Initialize session state to store conversation
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Function to handle user input
def handle_input():
    user_input = st.session_state.user_input
    if user_input.lower() == 'exit':
        st.session_state.conversation.append({"role": "chatbot", "content": "Thank you for using the service. Goodbye!"})
    elif user_input:
        # Store user input in the session state conversation list
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Get chatbot response
        response = chatbot.process_message(user_input)
        
        # Store chatbot response in the session state conversation list
        st.session_state.conversation.append({"role": "chatbot", "content": response})
        
    # Clear the input field after submission
    st.session_state.user_input = ""

# Set up the Streamlit UI
st.markdown("<h1 style='text-align: center;'>Travel Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Your personal travel assistant</h3>", unsafe_allow_html=True)

# Custom CSS for chat bubbles, headshots, and fixed input box at the bottom
st.markdown("""
    <style>
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 10px;
    }
    .user-bubble {
        background-color: #606060;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: left;
        width: fit-content;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .chatbot-bubble {
        background-color: #808080;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: left;
        width: fit-content;
        max-width: 70%;
    }
    .user-bubble, .chatbot-bubble {
        display: inline-block;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #fff;
        padding: 10px;
        box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Create a container for the conversation history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display the conversation history with chat bubbles and headshots
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='chatbot-bubble'>
                {message['content']}
            </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Input box at the bottom
with st.container():
    st.text_input(
        "Type your message here...", 
        key="user_input", 
        on_change=handle_input
    )
    st.markdown("</div>", unsafe_allow_html=True)
