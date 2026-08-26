#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from langchain.llms import GooglePalm
import google.generativeai as genai

api_key="AIzaSyAN8qDBM-rPoyB5exg83K62HH-d2iBi948"


# In[3]:


llm=GooglePalm(google_api_key=api_key,temperature=0.2)


# In[4]:


# poem=llm("write poem for my love for dosa")
# print(poem)


# In[5]:


agra_travel_qa = {
    "Q1": {
        "question": "What is the main attraction in Agra?",
        "answer": "The main attraction in Agra is the iconic Taj Mahal, a UNESCO World Heritage Site and one of the Seven Wonders of the World."
    },
    "Q2": {
        "question": "When is the best time to visit Agra?",
        "answer": "The best time to visit Agra is during the winter months, from October to March, when the weather is pleasant and suitable for sightseeing."
    },
    "Q3": {
        "question": "How can I reach Agra?",
        "answer": "Agra is well-connected by air, road, and rail. The nearest airport is Agra Airport (AGR), and there are regular trains and buses from major cities in India."
    },
    "Q4": {
        "question": "Apart from the Taj Mahal, what are other attractions in Agra?",
        "answer": "In addition to the Taj Mahal, Agra has other notable attractions like Agra Fort, Fatehpur Sikri, Itmad-ud-Daula's Tomb, and Mehtab Bagh."
    },
    "Q5": {
        "question": "What is the best time to visit the Taj Mahal?",
        "answer": "The best time to visit the Taj Mahal is early in the morning during sunrise or in the late afternoon during sunset to witness the monument bathed in soft, golden light."
    },
    "Q6": {
        "question": "Are there any specific rules or dress codes for visiting the Taj Mahal?",
        "answer": "Yes, visitors to the Taj Mahal are required to adhere to a dress code. Modest clothing is recommended, and certain items like tripods are not allowed inside the monument."
    },
    "Q7": {
        "question": "What are some recommended places to eat in Agra?",
        "answer": "Agra offers a variety of dining options. Some recommended places to eat include Peshawri, Joney's Place, and Dasaprakash."
    },
    "Q8": {
        "question": "Is Agra safe for tourists?",
        "answer": "Agra is generally safe for tourists, but it's advisable to take standard precautions. Avoid poorly lit areas at night, be cautious with belongings, and use authorized transportation services."
    },
    "Q9": {
        "question": "How much time should I plan for a visit to Agra?",
        "answer": "Agra can be explored in 2-3 days, allowing you to visit the major attractions like the Taj Mahal, Agra Fort, and Fatehpur Sikri at a comfortable pace."
    },
    "Q10": {
        "question": "Are there any cultural events or festivals in Agra that I should be aware of?",
        "answer": "Agra hosts various cultural events, and festivals throughout the year. The Taj Mahotsav, held in February, is a cultural extravaganza showcasing India's rich heritage."
    },
    "Q11": {
        "question": "Can you recommend some hotels in Agra?",
        "answer": "Certainly! Agra offers a range of accommodation options. Some notable hotels include The Oberoi Amarvilas, Tajview - IHCL SeleQtions, and Radisson Blu Agra Taj East Gate."
    },
    "Q12": {
        "question": "How can I book a hotel in Agra?",
        "answer": "You can book hotels in Agra through various online platforms like Booking.com, Agoda, or directly on the hotel's official website. It's advisable to book in advance, especially during peak tourist seasons."
    },
    "Q13": {
        "question": "What are the transportation options within Agra?",
        "answer": "Agra has a well-developed transportation system. You can use auto-rickshaws, cycle-rickshaws, and taxis for local travel. Additionally, many hotels offer transportation services, and there are app-based cab services like Uber and Ola available."
    },
    "Q14": {
        "question": "How can I travel from Agra to other cities in India?",
        "answer": "Agra is well-connected by train and road to major cities. There are also domestic flights from Agra Airport to some destinations. You can book train tickets through IRCTC and use long-distance buses or private cabs for road travel."
    },
    "Q15": {
        "question": "Is it easy to find transportation from the hotel to tourist attractions?",
        "answer": "Yes, most hotels in Agra offer transportation services for sightseeing. Additionally, auto-rickshaws, taxis, and cycle-rickshaws are readily available for short-distance travel within the city."
    }
}


# In[6]:


# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame.from_dict(agra_travel_qa, orient='index')

# Save the DataFrame to a CSV file
df.to_csv('agra_travel_qa.csv', index_label='Question_ID')

# Display the DataFrame
print(df)


# In[7]:


from datasets import load_dataset
from langchain.document_loaders.csv_loader import CSVLoader


# In[8]:


csv_file_path = "C:/Users/prakruthimadhav/Documents/travel chatbot/agra_travel_qa.csv"
loader= CSVLoader(file_path=csv_file_path,source_column="question")
data=loader.load()
data


# In[9]:


from langchain.embeddings import HuggingFaceInstructEmbeddings

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

e = instructor_embeddings.embed_query("What is your refund policy?")


# In[12]:


len(e)


# In[13]:


from langchain.vectorstores import FAISS

# Create a FAISS instance for vector database from 'data'
vectordb = FAISS.from_documents(documents=data,
                                 embedding=instructor_embeddings)

# Create a retriever for querying the vector database
retriever = vectordb.as_retriever(score_threshold = 0.7)


# In[14]:


rdocs = retriever.get_relevant_documents("how  far is the airport?")
rdocs


# In[15]:


from langchain.prompts import PromptTemplate

prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)


# In[17]:


chain('what is agra famous for?')


# In[ ]:




import streamlit as st

# Your application logic goes here

st.title("Agra Travel Chatbot")

# Create user input field
user_question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    # Call your chain or logic to get the answer
    answer = chain(user_question)
    st.write("Answer:", answer)
