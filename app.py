from typing import Generator
from utils import get_all_groq_model, validate_api_key, get_info, validate_uri
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

models = []

@st.cache_data
def get_text_models(api_key):
    models = get_all_groq_model(api_key=api_key)
    vision_audio = [model for model in models if 'vision' in model or 'whisper' in model]
    models = [model for model in models if model not in vision_audio]
    return models

# validating api_key
if not validate_api_key(api_key):
    st.sidebar.error("Enter valid API Key")
else:
    st.sidebar.success("API Key is valid")
    models = get_text_models(api_key)

model = st.sidebar.selectbox("Select Model", models)

if st.session_state.selected_model != model:
    st.session_state.messages = []
    st.session_state.selected_model = model


uri = st.sidebar.text_input("Enter SQL Database URI")
db_info = {'sql_dialect': '', 'tables': '', 'tables_schema': ''}
markdown_info = """
**SQL Dialect**: {sql_dialect}\n
**Tables**: {tables}\n
**Tables Schema**:
```sql
{tables_schema}
```
"""

if not validate_uri(uri):
    st.sidebar.error("Enter valid URI")
else:
    st.sidebar.success("URI is valid")
    db_info = get_info(uri)
    markdown_info = markdown_info.format(**db_info)
    with st.expander("SQL Database Info"):
        st.markdown(markdown_info)

system_prompt = f"""
You are an AI assistant specialized in generating optimized SQL queries based on user instructions. \
You have access to the database schema provided in a structured Markdown format. Use this schema to ensure \
correctness, efficiency, and security in your SQL queries.\

## SQL Database Info
{markdown_info}

---

## Query Generation Guidelines
1. **Ensure Query Validity**: Use only the tables and columns defined in the schema.
2. **Optimize Performance**: Prefer indexed columns for filtering, avoid `SELECT *` where specific columns suffice.
3. **Security Best Practices**: Always use parameterized queries or placeholders instead of direct user inputs.
4. **Context Awareness**: Understand the intent behind the query and generate the most relevant SQL statement.
5. **Formatting**: Return queries in a clean, well-structured format with appropriate indentation.
6. **Commenting**: Include comments in complex queries to explain logic when needed.

---

## Expected Output Format

The SQL query should be returned as a formatted code block:

```sql
-- Get all completed orders with user details
SELECT orders.id, users.name, users.email, orders.amount, orders.created_at
FROM orders
JOIN users ON orders.user_id = users.id
WHERE orders.status = 'completed'
ORDER BY orders.created_at DESC;
```

If the user's request is ambiguous, ask clarifying questions before generating the query.
"""

if model is not None and validate_uri(uri):
    client = Groq(
        api_key=api_key,
    )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
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
                    for m in st.session_state.messages
                ],
                max_tokens=3000,
                stream=True
            )

            # Use the generator function with st.write_stream
            with st.chat_message("SQL Assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response})