# TourTalker ✈️

Your AI-Powered Travel Assistant — Explore destinations, ask travel questions, and create personalized trip plans powered by AI.

## Project Overview

TourTalker is an intelligent travel assistant built with Streamlit that combines Retrieval-Augmented Generation (RAG) with Google's Gemini AI. It helps users explore travel destinations, ask questions about places, food, hotels, and attractions, and generate personalized trip itineraries based on their preferences.

The application uses a hybrid approach: it prioritizes information from a curated travel knowledge base while seamlessly integrating Gemini's general knowledge when the dataset is insufficient, ensuring comprehensive and helpful responses.

## Features

### 💬 Travel Q&A
- **Contextual Conversations**: Ask follow-up questions with conversation memory that maintains context across multiple turns
- **Hybrid Knowledge Retrieval**: Combines dataset information with AI general knowledge for comprehensive answers
- **Source Transparency**: Clear labeling of information sources ([DATASET] vs [AI KNOWLEDGE])
- **Smart Retrieval**: Uses contextualized queries that incorporate previous questions for better relevance

### 🗺️ Trip Planner
- **Personalized Itineraries**: Generate day-by-day trip plans based on user preferences
- **Destination Recommendations**: AI suggests destinations based on budget, interests, travel style, and duration
- **Budget Planning**: Detailed budget breakdown in INR with budget assessment
- **Flexible Options**: Supports custom destinations or AI-recommended ones
- **Interest-Based Planning**: Tailors activities to user interests (beaches, mountains, history, food, etc.)

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini (gemini-3.6-flash)
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Framework**: LangChain
- **Language**: Python 3.11+

## Project Structure

```
TourTalker-master/
├── main.py                 # Streamlit frontend application
├── helper.py               # Backend logic (RAG, embeddings, LLM integration)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API key)
├── .gitignore              # Git ignore rules
├── Procfile                # Render deployment configuration
├── combined_dataset_travel.csv  # Travel knowledge base
├── faiss_index/            # Pre-built FAISS vector database
│   ├── index.faiss
│   └── index.pkl
└── venv/                   # Virtual environment (not committed)
```

## How It Works

### Architecture Flow

```
User Question/Preferences
        ↓
Contextualized Query (for follow-ups)
        ↓
FAISS Vector Database Retrieval
        ↓
Relevant Travel Context + User Input
        ↓
Google Gemini LLM (Hybrid Approach)
        ↓
Response with Source Labels
```

### Travel Q&A Workflow

1. **User Input**: User asks a travel question
2. **Contextualization**: For follow-up questions, the system builds a contextualized query using the last 3 user questions to maintain topic context
3. **Retrieval**: FAISS searches the vector database for relevant travel information
4. **Hybrid Generation**: Gemini processes the retrieved context and user question:
   - Prioritizes dataset information when relevant
   - Uses general knowledge when dataset is insufficient
   - Labels sources clearly ([DATASET] or [AI KNOWLEDGE])
5. **Conversation Memory**: Responses are stored in session state for context in follow-ups

### Trip Planner Workflow

1. **User Preferences**: User provides starting city, destination (optional), budget, duration, travelers, style, and interests
2. **Context Retrieval**: FAISS searches for relevant travel information about the destination
3. **Hybrid Planning**: Gemini generates a personalized itinerary:
   - Uses dataset information when available
   - Supplements with general knowledge when needed
   - Respects user's explicitly chosen destination
   - Recommends destinations only when not specified
4. **Budget Analysis**: Provides detailed budget breakdown in INR with assessment against user's budget range

## Installation and Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Google Gemini API key

### Steps

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd TourTalker-master
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
   - Create a `.env` file in the project root
   - Add your Google Gemini API key:
```
GOOGLE_API_KEY=your_api_key_here
```

5. **Ensure FAISS index exists**
   - The `faiss_index/` folder should contain `index.faiss` and `index.pkl`
   - If missing, run the vector database creation (see below)

### Creating the Vector Database (if needed)

If the FAISS index is not present, create it using:
```python
from helper import create_vector_db
create_vector_db()
```

This will process `combined_dataset_travel.csv` and generate the vector database.

## How to Run Locally

1. **Activate your virtual environment**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Run the Streamlit application**
```bash
streamlit run main.py
```

3. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## Environment Variables

The application requires the following environment variable:

- `GOOGLE_API_KEY`: Your Google Gemini API key for LLM access

**Important**: Never commit your `.env` file or expose your API key in public repositories. The `.gitignore` file is configured to exclude `.env` by default.

## Limitations

- The travel knowledge base is limited to the destinations and information present in the dataset
- Responses depend on the quality and coverage of the FAISS vector database
- The application requires an active internet connection for Gemini API calls
- Performance depends on the available system memory (FAISS index is cached after first load)

## Future Improvements

- Expand the travel knowledge base with more destinations and information
- Add support for multiple languages
- Implement user authentication and personalized trip history
- Add real-time flight and hotel price integration
- Support for image-based destination recommendations
- Mobile-responsive design improvements

## License

This project is for educational and demonstration purposes.

## Acknowledgments

- Travel dataset used for knowledge base
- Google Gemini API for LLM capabilities
- HuggingFace for sentence transformer embeddings
- Streamlit for the frontend framework
