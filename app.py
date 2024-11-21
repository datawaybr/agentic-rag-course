import time

import streamlit as st

import chatbot


def response_generator():

    response = chatbot.get_response(st.session_state.messages)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("DWAY Copilot")

if "messages" not in st.session_state:
    base_prompt = {
        "role": "system",
        "content": "You are an agent created to be a Machine Learning copilot to create Python code. To help you create the requested code you have to look for documentation on the vector database. If the vector database doesn't have the information needed, then you have to use the docs_search tool to gather more useful information for your answer.",
    }
    st.session_state.messages = [base_prompt]


for message in st.session_state.messages:

    if (message["role"] == "assistant" and message["content"] != None) or message[
        "role"
    ] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    st.session_state.messages.append({"role": "assistant", "content": response})
