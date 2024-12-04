import os
from langchain_chroma import Chroma
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.chat_models import ChatOllama



class LLM():

    def __init__(self, file_path):
        self.file_path = file_path

    # ingest pdf
    local_path = ".\\uploaded_files\\rag_eg.pdf"
    if local_path:
        loader = UnstructuredPDFLoader(file_path=local_path)
        data = loader.load()
    else:
        print("file not found")


    # splitting chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="myRAG"
        ) 


    # retrieval
    model = OllamaLLM(model="llama3")
    llm = ChatOllama(model="llama3")

    def process_and_store(self):
        # Ingest pdf or other file types based on file extension
        if os.path.exists(self.file_path):
            file_extension = os.path.splitext(self.file_path)[1].lower()
            if file_extension == '.pdf':
                loader = UnstructuredPDFLoader(file_path=self.file_path)
            # Add more loaders here for txt, docx if needed
            else:
                print(f"Unsupported file type: {file_extension}")
                return

            # Load document data
            data = loader.load()

            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

            # Embeddings and vector database setup
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="myRAG"
            )

            # Store or update the vector database as needed
            print(f"File '{self.file_path}' processed and stored in vector database.")

        else:
            print("File not found")
