import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. "
        "Please set it in your .env file."
    )

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0.2,
    convert_system_message_to_human=True
)

# Initialize embeddings
instructor_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Set paths using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent
csv_file_path = BASE_DIR / "combined_dataset_travel.csv"
vectordb_file_path = BASE_DIR / "faiss_index"

# Cache for FAISS vector database to avoid repeated disk I/O
_cached_vectordb = None

def get_vector_db():
    """Load and cache the FAISS vector database to avoid repeated disk I/O."""
    global _cached_vectordb
    if _cached_vectordb is None:
        if not vectordb_file_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {vectordb_file_path}. "
                f"Please run create_vector_db() first to generate the index."
            )
        _cached_vectordb = FAISS.load_local(str(vectordb_file_path), instructor_embeddings, allow_dangerous_deserialization=True)
    return _cached_vectordb

def create_vector_db():
    """Create and save the FAISS vector database from the CSV dataset."""
    # Check if CSV file exists
    if not csv_file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_file_path}. "
            f"Please ensure combined_dataset_travel.csv exists in the project directory."
        )
    
    # Load data from FAQ sheet
    loader = CSVLoader(file_path=str(csv_file_path), source_column="question")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(str(vectordb_file_path))
    print(f"Vector database created and saved to {vectordb_file_path}")


def get_qa_chain():
    """Load the vector database and create the QA chain."""
    # Check if FAISS index exists
    if not vectordb_file_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {vectordb_file_path}. "
            f"Please run create_vector_db() first to generate the index."
        )
    
    # Load the vector database from cache
    vectordb = get_vector_db()

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "answer" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


def get_qa_chain_with_history(current_question, conversation_history="", contextualized_query=None):
    """Load the vector database and perform QA with conversation history and contextualized retrieval."""
    # Check if FAISS index exists
    if not vectordb_file_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {vectordb_file_path}. "
            f"Please run create_vector_db() first to generate the index."
        )
    
    # Load the vector database from cache
    vectordb = get_vector_db()

    # Use contextualized query for retrieval if provided, otherwise use current question
    retrieval_query = contextualized_query if contextualized_query else current_question

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Retrieve relevant documents using the contextualized query
    relevant_docs = retriever.get_relevant_documents(retrieval_query)

    # Combine retrieved context
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Build conversation history string if provided
    history_context = ""
    if conversation_history:
        history_context = f"\n\nCONVERSATION HISTORY:\n{conversation_history}\n"

    # Build the complete prompt
    prompt = f"""You are a hybrid AI travel assistant with access to a travel knowledge base and general knowledge.

INSTRUCTIONS:
1. PRIORITIZE the retrieved travel dataset context when it contains relevant, specific information.
2. If the dataset context is insufficient, irrelevant, or doesn't contain the answer, use your general knowledge to provide a helpful response.
3. Clearly label the source of information:
   - Use [DATASET] for information retrieved from the travel knowledge base
   - Use [AI KNOWLEDGE] for information from your general knowledge
4. When both sources are useful, combine them and clearly label each part.
5. Do NOT say "I don't know" if you can provide a helpful answer using your general knowledge.
6. Use the conversation history to understand the context of follow-up questions.{history_context}

RETRIEVED TRAVEL KNOWLEDGE:
{context}

QUESTION: {current_question}

Provide a comprehensive answer with clear source labels."""

    # Generate response using LLM
    response = llm.invoke(prompt)

    return {"result": response.content, "source_documents": relevant_docs}


def plan_trip(starting_city, destination, budget, travel_days, interests, 
              num_travelers, travel_style):
    """Generate a personalized trip plan based on user preferences."""
    # Check if FAISS index exists
    if not vectordb_file_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {vectordb_file_path}. "
            f"Please run create_vector_db() first to generate the index."
        )
    
    # Load the vector database from cache
    vectordb = get_vector_db()
    
    # Create a search query based on user preferences
    search_query = f"travel to {destination if destination else starting_city} " \
                   f"budget {budget} {travel_style} style " \
                   f"interests: {', '.join(interests)}"
    
    # Retrieve relevant context from the database
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})
    relevant_docs = retriever.get_relevant_documents(search_query)
    
    # Combine retrieved context
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Create trip planning prompt
    trip_prompt = f"""You are an expert travel planner. Create a personalized {travel_days}-day trip itinerary based on the following user preferences and retrieved travel information.

USER PREFERENCES:
- Starting City: {starting_city}
- Destination: {destination if destination else 'To be recommended based on interests'}
- Budget: {budget}
- Number of Travelers: {num_travelers}
- Travel Days: {travel_days}
- Interests: {', '.join(interests)}
- Travel Style: {travel_style}

RETRIEVED TRAVEL INFORMATION FROM DATASET:
{context}

INSTRUCTIONS:
1. DESTINATION HANDLING:
   - If the user has explicitly provided a destination ("{destination}"), you MUST create the itinerary specifically for that destination. Never change, replace, or recommend a different destination.
   - If the destination field is empty or not provided, then recommend the most suitable destination based on their starting city, interests, budget, travel days, and travel style.
2. PRIORITIZE the retrieved travel dataset when it contains relevant, specific information.
3. If the dataset context is insufficient, irrelevant, or doesn't contain the answer, use your general knowledge to provide a helpful response.
4. Create a detailed day-by-day itinerary with specific activities and places.
5. Provide a budget breakdown for accommodation, food, transport, and activities.
6. Suggest places and activities matching their interests.
7. Clearly label the source of information:
   - Use [DATASET] for information retrieved from the travel knowledge base
   - Use [AI KNOWLEDGE] for information from your general knowledge
8. When both sources are useful, combine them and clearly label each part.
9. Do NOT say "I don't know" if you can provide a helpful answer using your general knowledge.
10. Be realistic about the budget and travel style.
11. ALL monetary amounts must be in Indian Rupees (₹). Do not use USD or any other currency.
12. Calculate the total estimated cost and compare it against the user's budget range. If the trip exceeds their budget, clearly state this at the end with a warning.
13. Keep the response reasonably concise. Avoid unnecessarily long paragraphs or excessive detail.
14. Format the budget breakdown cleanly with proper spacing and no merged text.

Format your response as:

Recommended Destination:
[Destination name and why it matches their preferences]

Day 1:
[Activities and places]

Day 2:
[Activities and places]

[Continue for all days...]

Estimated Budget Breakdown:
- Accommodation: ₹[amount]
- Food: ₹[amount]
- Transport: ₹[amount]
- Activities: ₹[amount]
- Total Estimated Budget: ₹[amount]

Budget Assessment:
[State whether this fits within the user's budget range of {budget}. If it exceeds, clearly warn them.]

Travel Tips:
[Relevant tips based on their preferences]"""
    
    # Generate response using LLM
    response = llm.invoke(trip_prompt)
    
    return response.content, relevant_docs

if __name__ == "__main__":
    try:
        print("Creating vector database...")
        create_vector_db()
        print("\nTesting QA chain...")
        chain = get_qa_chain()
        response = chain("What are the must-visit attractions in Delhi?")
        print("\nResponse:", response["result"])
    except Exception as e:
        print(f"Error: {e}")