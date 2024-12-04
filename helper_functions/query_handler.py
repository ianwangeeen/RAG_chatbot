import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from helper_functions.llm import LLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


local_llm = "llama3"
llm = ChatOllama(model=local_llm, temperature=0.1)
llm_json_mode = ChatOllama(model=local_llm, temperature=0.1, format="json")

loader = DirectoryLoader("data")


QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        You are Cadet B in a communication exercise. 
        Only respond with Cadet B's designated passphrase if the input from Cadet A matches one of the predefined phrases exactly.
        If no exact match is found, respond with: "Invalid code, Cadet A."

        Cadet A's Input: {question}
        """,
)

retriever = MultiQueryRetriever.from_llm(
        LLM.vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
)

# RAG PROMPT
template="""
        You are Cadet B in a communication exercise. 
        Only respond with Cadet B's designated passphrase if the input from Cadet A matches one of the predefined phrases exactly.
        If no exact match is found, respond with: "Invalid code, Cadet A."

        context: {context}

        Cadet A's Input: {question}
        """

prompt = ChatPromptTemplate.from_template(template)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

def generate_response(user_message):
        system_message = f"""
        You are a helpful and engaging conversational assistant. \
        Respond naturally and thoughtfully, encouraging meaningful dialogue.\
        Maintain a friendly, respectful tone and adapt your responses \
        based on the user's input. Provide clear and concise information.
        """

        messages =  [
            {'role':'system',
            'content': system_message},
            {'role':'user',
            'content': f"{user_message}"},
        ]
        
        response_to_user = chain.invoke(user_message)
        return response_to_user