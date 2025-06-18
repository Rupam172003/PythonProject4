# Phase 1 libraries
import os
import warnings
import logging
import tempfile
import hashlib

import streamlit as st

# Phase 2 libraries
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask Chatbot!')

# PDF upload section
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Setup session state for messages and vectorstore
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None

# Process uploaded PDF
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    new_file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Only process if it's a new file
    if st.session_state.file_hash != new_file_hash:
        with st.spinner("Processing PDF..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Create vector store
            try:
                loaders = [PyPDFLoader(tmp_path)]
                index = VectorstoreIndexCreator(
                    embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
                    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                ).from_loaders(loaders)

                st.session_state.vectorstore = index.vectorstore
                st.session_state.file_hash = new_file_hash
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
    else:
        st.info("Using previously processed PDF")

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Chat input
prompt = st.chat_input('Ask a question about your PDF')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Phase 2 setup
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. 
        Answer the user's question based on the provided context. 
        If you don't know the answer, say you don't know. 
        Question: {user_prompt}""")

    model ="meta-llama/llama-4-scout-17b-16e-instruct"

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    # Process question
    if st.session_state.vectorstore is None:
        st.error("Please upload a PDF file first")
    else:
        try:
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=False
            )

            result = chain({"query": prompt})
            response = result["result"]

            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append(
                {'role': 'assistant', 'content': response})
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")