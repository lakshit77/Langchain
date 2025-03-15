import streamlit as st
import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Set page title
st.set_page_config(page_title="Document Q&A Assistant")

# Define functions (same as before)def load_documents(uploaded_files):
def load_documents(uploaded_files):
    """Load documents from uploaded PDF files."""
    documents = []
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    return documents

def process_documents(documents):
    """Split documents into chunks and create a vector store."""
# Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

# Create embeddings and store in vector database
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

def create_qa_chain(vector_store):
    """Create an enhanced QA chain with better prompting."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

# Create a custom prompt template
    template = """
    You are a helpful assistant that answers questions based on provided documents.

    Context information from documents:
    {context}

    Question: {question}

    Answer the question based only on the provided context. If you don't know the answer or cannot find it in the context, say "I couldn't find this information in the provided documents." Include specific details and cite the sources of information.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Create QA chain with custom prompt
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# Streamlit interface
st.title("Document Q&A Assistant")
st.write("Upload your documents and ask questions about them!")

# API key input
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# File uploader
uploaded_files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)

if uploaded_files and api_key:
# Process buttonif st.button("Process Documents"):
        with st.spinner("Processing documents..."):
# Load and process documents
            documents = load_documents(uploaded_files)
            st.session_state.vector_store = process_documents(documents)
            st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store)
            st.success(f"Successfully processed {len(uploaded_files)} documents!")

# Check if documents have been processedif 'qa_chain' in st.session_state:
# Question input
        question = st.text_input("Ask a question about your documents:")
        if question:
            with st.spinner("Thinking..."):
# Get answer
                result = st.session_state.qa_chain({"query": question})

# Display answer
                st.write("### Answer:")
                st.write(result["result"])

# Display sources
                st.write("### Sources:")
                for i, doc in enumerate(result["source_documents"]):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    st.write(f"Source {i+1}: {os.path.basename(source)}, Page {page}")
