import streamlit as st
from rag_engine import RAGengine
import time
import os

#page-configuration
st.set_page_config(page_title="RAG research assistant",layout="wide")#shows the name in the browser tab
st.title("Document Based Chatbot")

#--sidebar(file upload)
st.sidebar.header("Document Ingestion")#creates a space in the left side (portion) of the web app
uploaded_file = st.sidebar.file_uploader("Upload pdf",type="pdf")

if "messages" not in st.session_state:
    st.session_state.messages=[]

if st.sidebar.button("clear chat"):
    st.session_state.messages=[]
    st.rerun()

if uploaded_file:
    if("current_pdf" not in st.session_state or st.session_state.current_pdf!=uploaded_file.name):
        with st.sidebar.spinner("Ingesting the document...."):
            with open("temp.pdf","wb") as f:
                f.write(uploaded_file.getbuffer())

        
            st.session_state.engine = RAGengine(pdf_path="temp.pdf")
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.messages = []
            st.sidebar.success("Document Ingested")

#chat-display logic 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#user_input
if prompt:= st.chat_input("Ask anything "):
    if "engine" not in st.session_state:
        st.warning("Please insert a pdf first!")
    else:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    #connecting the rag engine

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = st.session_state.messages[-6:]
                response = st.session_state.engine.generate_response(prompt,history)
                placeholder = st.empty()
                full_response = ""
                for word in response.split():
                    full_response+=word+" "
                    placeholder.markdown(full_response)
                    time.sleep(0.02)

        st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})



