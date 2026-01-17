import streamlit as st
from src.loader import DocumentLoader
from src.rag_engine import RAGEngine

# Page Config
st.set_page_config(page_title="Dalil-RAG", layout="wide")

# Title and Description
st.title("Dalil-RAG")
st.markdown("Chat with your data using a simple RAG.")

# Sidebar for Setup
with st.sidebar:
    st.header("Settings")
    # We need a token to use Hugging Face's free API
    hf_token = st.text_input(
        "Enter Hugging Face Token",
        type="password",
    )

    st.divider()

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"]
    )

    if st.button("Process Documents"):
        if not hf_token:
            st.error("Please enter a Hugging Face Token first.")
        elif not uploaded_files:
            st.error("Please upload at least one file.")
        else:
            with st.spinner("Starting now!"):
                # Initialize Engine
                rag_engine = RAGEngine(hf_token=hf_token)

                # Load Docs
                docs = DocumentLoader.load_files(uploaded_files)

                # Build index
                rag_engine.build_index(docs)

                # Save to session state
                st.session_state.rag_engine = rag_engine
                st.success("Documents Processed! You can now chat.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ready to chat? Type your message here..."):
    # Check if engine is ready
    if "rag_engine" not in st.session_state:
        st.error("Please upload and process documents first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            query_engine = st.session_state.rag_engine.get_query_engine()
            response_stream = query_engine.query(prompt)

            # Stream the response
            response_text = st.write_stream(response_stream.response_gen)

            # Save assistant response to history
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
