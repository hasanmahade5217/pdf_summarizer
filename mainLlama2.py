# Import necessary libraries
import streamlit as st # UI library

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter # Split the PDF into chunks
from langchain_community.vectorstores import FAISS # Vector store 

import google.generativeai as genai #import google gemini API
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Google GENAI Embeddings

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate # langchain prompt templates
from langchain.chains.question_answering import load_qa_chain # lang chain  QA chain
from langchain.chains import RetrievalQA

import os 
from dotenv import load_dotenv

load_dotenv()



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    ollama_emb = OllamaEmbeddings(base_url="http://10.37.156.165:7860", model="llama2:latest")
    vector_store = FAISS.from_texts(text_chunks, embedding=ollama_emb)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # model creation: Llama2
    llm_model = Ollama(
        base_url= "http://10.37.156.165:7860",
        model= "llama2:latest",
        temperature=0.3,

    )
    
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm_model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    ollama_emb = OllamaEmbeddings(base_url="http://10.37.156.165:7860", model="llama2:latest")
    
    new_db = FAISS.load_local("faiss_index", ollama_emb, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Llama2 using Ollama")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

