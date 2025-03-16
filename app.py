from typing import Generator
from utils import validate_api_key, get_info, validate_uri, extract_code_blocks
from langchain_community.utilities import SQLDatabase
from var import system_prompt, markdown_info, query_output, groq_models
import streamlit as st
from groq import Groq

st.set_page_config(layout="wide")

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

st.markdown("# SQL Chat")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password")

# validating api_key
if not validate_api_key(api_key):
    st.sidebar.error("Enter valid API Key")
    model = st.sidebar.selectbox("Select Model", groq_models, disabled=True)
else:
    st.sidebar.success("API Key is valid")
    model = st.sidebar.selectbox("Select Model", groq_models, index=0)

if st.session_state.selected_model != model:
    st.session_state.messages = []
    st.session_state.selected_model = model

uri = st.sidebar.text_input("Enter SQL Database URI")

if not validate_uri(uri):
    st.sidebar.error("Enter valid URI")
else:
    st.sidebar.success("URI is valid")
    db_info = get_info(uri)
    markdown_info = markdown_info.format(**db_info)
    with st.expander("SQL Database Info"):
        st.markdown(markdown_info)
    system_prompt = system_prompt.format(markdown_info = markdown_info)

if validate_api_key(api_key) and validate_uri(uri):
    client = Groq(
        api_key=api_key,
    )

    db = SQLDatabase.from_uri(uri)
    
    avatar = {"user": '👨‍💻', "assistant": '🤖', "executor": '🛢'}

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=avatar[message["role"]]):
            st.markdown(message["content"])


    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=avatar["user"]):
            st.markdown(prompt)

        # Fetch response from Groq API
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[{
                        "role": "system",
                        "content": system_prompt
                    },
                ]+
                [
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages[-8:]
                ],
                max_tokens=3000,
                stream=True
            )

            # Use the generator function with st.write_stream
            with st.chat_message("SQL Assistant", avatar=avatar["assistant"]):
                chat_responses_generator = generate_chat_responses(chat_completion)
                llm_response = st.write_stream(chat_responses_generator)

            with st.chat_message("SQL Executor", avatar=avatar["executor"]):
                query = extract_code_blocks(llm_response)
                result = db.run(query[0])
                query_response = st.write(query_output.format(result=result))

        except Exception as e:
            st.error(e, icon="🚨")

        if len(str(result)) > 1000:
            query_output_truncated = query_output.format(result=result)[:500]+query_output.format(result=result)[-500:]
        else:
            query_output_truncated = query_output.format(result=result)

        # Append the llm response to session_state.messages
        if isinstance(llm_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": llm_response + query_output_truncated})
        else:
            # Handle the case where llm_response is not a string
            combined_response = "\n".join(str(item) for item in llm_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response + query_output_truncated})

    st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.messages.clear())

else:
    st.error("Please enter valid Groq API Key and URI in the sidebar.")