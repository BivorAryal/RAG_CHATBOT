import streamlit as st
from utils import load_pdf 
from connect_memory import rag_chain, retrievers

st.title("🎯 PM Job Interview Coach")

user_query = st.text_area("Enter your prompt: ", placeholder="Ask Anything!")
ask_question = st.button("ASK PM Expert")

if ask_question and user_query.strip():  # Check if query is not empty
    st.chat_message("user").write(user_query)

    # RAG Pipeline
    retrieved_docs = retrievers.get_relevant_documents(user_query)  
    response = rag_chain.invoke(user_query)  
    st.chat_message("AI Lawyer").write(response)  
else:
    st.error("Enter a valid question!")