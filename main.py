import streamlit as st
from helper import get_qa_chain, create_vector_db, plan_trip, get_qa_chain_with_history

# Set page configuration with sky blue theme
st.set_page_config(
    page_title="TourTalker",
    page_icon="✈️"
)

# Custom CSS for sky blue theme
st.markdown("""
<style>
    .stApp {
        background-color: #E6F3FF;
    }
    /* Main text and headings */
    h1, h2, h3, h4, h5, h6, .stTitle, .stMarkdown {
        color: #1E3A5F !important;
    }
    /* Secondary text */
    p, .stText, label {
        color: #374151 !important;
    }
    /* Tab text */
    .stTabs [data-baseweb="tab"] {
        color: #1E3A5F !important;
    }
    /* Input placeholder */
    .stTextInput>div>div>input::placeholder {
        color: #6B7280 !important;
    }
    /* Input text */
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        color: #1F2937 !important;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4A90E2;
        color: white;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0 1rem;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #B8D4E8;
        color: #1E3A5F;
        border-bottom: 3px solid #4A90E2;
    }
</style>
""", unsafe_allow_html=True)

# Function to get the chatbot response
def get_bot_response(user_input):
    try:
        chain = get_qa_chain()
        response = chain(user_input)
        return response
    except FileNotFoundError as e:
        return {"result": f"Error: {str(e)}", "source_documents": []}
    except ValueError as e:
        return {"result": f"Error: {str(e)}", "source_documents": []}
    except Exception as e:
        return {"result": f"An error occurred: {str(e)}", "source_documents": []}

# Streamlit UI
def main():
    st.title("TourTalker ✈️")
    st.markdown("### Your AI-Powered Travel Assistant 🌍")
    st.markdown("Explore destinations, ask travel questions, and create personalized trip plans powered by AI.")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Travel Q&A", "🗺️ Trip Planner"])
    
    # Tab 1: Travel Q&A (existing functionality with conversation memory)
    with tab1:
        st.markdown("Ask me anything about travel!")
        
        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Conversation History")
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(user_msg)
                with st.chat_message("assistant"):
                    st.write(bot_msg)
                st.divider()
        
        # Clear chat button
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        # User input text box
        user_input = st.text_input("", placeholder="Ask about destinations, food, hotels, attractions, or travel tips...")
        
        if st.button("Ask"):
            if not user_input.strip():
                st.warning("Please enter a question.")
                return
            
            # Build conversation history string
            conversation_history = ""
            if st.session_state.chat_history:
                for prev_user, prev_bot in st.session_state.chat_history[-3:]:  # Last 3 exchanges
                    conversation_history += f"User: {prev_user}\nAssistant: {prev_bot}\n"
            
            # Build contextualized retrieval query using previous user messages
            contextualized_query = user_input
            if st.session_state.chat_history:
                # Get the last 3 user questions for context
                recent_user_questions = []
                for prev_user, prev_bot in st.session_state.chat_history[-3:]:
                    recent_user_questions.append(prev_user)
                # Combine recent questions with current question for better retrieval
                contextualized_query = " ".join(recent_user_questions) + " " + user_input
            
            # Get chatbot response with history and contextualized query
            with st.spinner("Thinking..."):
                try:
                    bot_response = get_qa_chain_with_history(
                        current_question=user_input,
                        conversation_history=conversation_history,
                        contextualized_query=contextualized_query
                    )
                    response_text = bot_response["result"]
                except FileNotFoundError as e:
                    response_text = f"Error: {str(e)}"
                except ValueError as e:
                    response_text = f"Error: {str(e)}"
                except Exception as e:
                    response_text = f"An error occurred: {str(e)}"
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, response_text))
            
            # Display the new message
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                st.write(response_text)
    
    # Tab 2: Trip Planner (new feature)
    with tab2:
        st.markdown("### Plan Your Perfect Trip")
        st.markdown("Tell us your preferences and we'll create a personalized itinerary!")
        
        with st.form("trip_planner_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                starting_city = st.text_input("Starting City*", placeholder="e.g., Delhi")
                destination = st.text_input("Destination (Optional)", placeholder="Leave blank for recommendations")
                budget = st.selectbox("Budget (INR)", ["Budget (₹15,000-30,000)", "Moderate (₹30,000-75,000)", "Luxury (₹75,000+)"])
                travel_days = st.slider("Number of Travel Days", min_value=1, max_value=14, value=3)
            
            with col2:
                num_travelers = st.number_input("Number of Travelers", min_value=1, max_value=20, value=1)
                travel_style = st.selectbox("Travel Style", ["Budget", "Moderate", "Luxury"])
            
            interests = st.multiselect(
                "Travel Interests",
                ["Beaches", "Mountains", "Adventure", "History", "Food", "Wildlife", "Culture", "Nightlife", "Shopping", "Nature"],
                default=["History", "Food"]
            )
            
            submitted = st.form_submit_button("Generate Trip Plan")
            
            if submitted:
                # Validate inputs
                if not starting_city.strip():
                    st.error("Please enter a starting city.")
                    return
                
                if not interests:
                    st.error("Please select at least one travel interest.")
                    return
                
                # Generate trip plan
                with st.spinner("Creating your personalized trip plan..."):
                    try:
                        trip_plan, source_docs = plan_trip(
                            starting_city=starting_city,
                            destination=destination,
                            budget=budget,
                            travel_days=travel_days,
                            interests=interests,
                            num_travelers=num_travelers,
                            travel_style=travel_style
                        )
                        
                        st.success("Trip plan generated!")
                        st.markdown(trip_plan)
                        
                        # Show sources
                        with st.expander("View Information Sources"):
                            st.write("Information retrieved from travel dataset:")
                            for i, doc in enumerate(source_docs, 1):
                                st.write(f"{i}. {doc.page_content[:200]}...")
                    
                    except FileNotFoundError as e:
                        st.error(f"Error: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

