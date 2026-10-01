import streamlit as st

from api_tool_agent.agent import (
    get_final_text,
    run_agent,
)
from api_tool_agent.api.client import upload_document

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Agent API Hub",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Agent API Hub")

st.caption("A custom AI agent powered by Gemini, LangChain, " "and API-wrapped tools.")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Agent Tools")

    st.markdown("""
        **Document Tools**
        - Upload PDF/TXT
        - Ask questions about documents

        **Weather Tool**
        - Current weather
        - Temperature
        - Humidity
        - Wind speed

        **Titanic Tool**
        - ML-based fare prediction
        """)

    st.divider()

    # -----------------------------------------------------
    # Document Upload
    # -----------------------------------------------------

    st.subheader("Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
    )

    if uploaded_file is not None:

        if st.button(
            "Upload Document",
            use_container_width=True,
        ):
            try:
                result = upload_document(
                    filename=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                )

                st.session_state.document_id = result["document_id"]

                st.session_state.document_name = result["filename"]

                st.success(f"Uploaded successfully: " f"{uploaded_file.name}")

                st.rerun()

            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    # -----------------------------------------------------
    # Active Document
    # -----------------------------------------------------

    if st.session_state.document_id:

        st.divider()

        st.subheader("Active Document")

        st.write(f"**{st.session_state.document_name}**")

        st.caption(f"Document ID: " f"{st.session_state.document_id}")

        if st.button(
            "Remove Document",
            use_container_width=True,
        ):
            st.session_state.document_id = None
            st.session_state.document_name = None

            st.rerun()

    # -----------------------------------------------------
    # Clear Chat
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------

user_message = st.chat_input("Ask about weather, Titanic fares, or your document...")


if user_message:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_message)

    history = list(st.session_state.messages)

    # -----------------------------------------------------
    # Add active document context
    # -----------------------------------------------------

    agent_message = user_message

    if st.session_state.document_id:

        agent_message = f"""
There is currently an uploaded document available.

Document ID:
{st.session_state.document_id}

Document filename:
{st.session_state.document_name}

User message:
{user_message}

If the user is referring to the uploaded document,
use ask_document_tool with the document ID above.

Do not ask the user for a local file path because
the document has already been uploaded.
"""

    # -----------------------------------------------------
    # Run Agent
    # -----------------------------------------------------

    try:

        with st.chat_message("assistant"):

            with st.spinner("Agent is working..."):

                result = run_agent(
                    message=agent_message,
                    history=history,
                )

                response = get_final_text(result)

            st.markdown(response)

    except Exception as exc:

        response = "An error occurred while processing " f"your request: {exc}"

        with st.chat_message("assistant"):
            st.error(response)

    # -----------------------------------------------------
    # Save Conversation
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
