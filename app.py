import streamlit as st
import os
from rag_pipeline import MovieRAG
from dotenv import load_dotenv

load_dotenv()

# Setup Streamlit page configuration
st.set_page_config(page_title="Movie Knowledge Chatbot", page_icon="🎬", layout="wide")

# Custom CSS for ChatGPT-like UI
st.markdown("""
<style>
    .reportview-container {
        background: #f4f4f9;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 800px;
        margin: auto;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #e6f7ff;
    }
    .stChatMessage.assistant {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎬 TMDB Movie Assistant")
    st.markdown("Ask anything about the TMDB 5000 movies!")
    st.markdown("---")
    st.markdown("**Example Questions:**")
    st.markdown("- Who directed Spider Man 3?")
    st.markdown("- What are some good sci-fi movies?")
    st.markdown("- Tell me about movies featuring Leonardo DiCaprio.")
    st.markdown("---")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.caption("Powered by Gemini and FAISS Vector Search")

st.title("🎬 Movie Knowledge Chatbot")

# Initialize RAG Pipeline in session state to avoid reloading
if 'rag' not in st.session_state:
    try:
        with st.spinner("Initializing knowledge base (this may take a few minutes on the first run)..."):
            rag = MovieRAG()
            # This extracts data and builds the FAISS index if missing
            rag.auto_setup()
            
            if not rag.load_index():
                st.error("Vector database initialization failed! Check terminal logs.")
                st.stop()
            st.session_state.rag = rag
            # st.success("Database loaded successfully!") # Removing success message to keep UI clean like ChatGPT
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        st.stop()

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your movie expert. Ask me about any movie from the TMDB database!"}
    ]

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about a movie... (e.g., 'Who directed Inception?')"):
    
    if not os.getenv("GEMINI_API_KEY"):
         st.error("GEMINI_API_KEY environment variable not found. Please set it in a .env file.")
         st.stop()
         
    # Display user query in chat
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Check if this is a general greeting or out of scope, but we rely on the prompt to enforce constraints.
    with st.chat_message("assistant"):
        with st.spinner("Searching the TMDB knowledge base..."):
            try:
                # Generate answer using the vector DB and LLM
                response = st.session_state.rag.generate_answer(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Sorry, an error occurred while processing your query: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
