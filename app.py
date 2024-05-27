import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from llm_log import LoggingChatOpenAI
from langchain_openai import ChatOpenAI
from final_workflow import extract_final_output
from chroma_db import generate_data_store
from pathlib import Path
import json
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

WORKING_DIR = Path.cwd()
db_path = WORKING_DIR / 'chroma2.0'

openai_api_key = os.getenv("OPENAI_API_KEY")



def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def retrieve_data(llm, openai_api_key, CHROMA_PATH, user_query):
    embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
    print("embedding function", embedding_function)
    # exit()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    print("retriver",retriever)

    message = """
    You will be asked a question about which functions to call to achieve the stated purpose and in what sequence. Return the name of the functions and their descriptions and input directly drawn from the input strictly without using your own intelligence(if no activity found, just return 'No Activity Found' for specified purpose) in the following JSON Format without any further processing :

    Question:
    {question}

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([("human", f"{message}")])
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    
    response = rag_chain.invoke(user_query)

    return response

def main():

    if not db_path.is_dir():
        generate_data_store()
        print("Created DB-------------------------------------------------")


    st.title("RAG Retriever")

    user_query = st.text_input("Enter your query:")

    if st.button("Retrieve Workflow"):
        if user_query:
            with st.spinner("Integrating LLM..."):
                print("we are here----------------------------------------------------------------------------")
                CHROMA_PATH = "chroma2.0"

                llm_retrieval = ChatOpenAI(model="gpt-4-1106-preview", api_key=openai_api_key)

            with st.spinner("Retrieving data from ChromaDB..."):
                response = retrieve_data(llm_retrieval, openai_api_key, CHROMA_PATH, user_query)


            response_data = response.content
            response_data = response_data.replace("json", "").replace("```", "")

            data = json.loads(response_data)


            with open('output.json', 'w') as json_file:
                json.dump(data, json_file, indent=2)

            json_data = load_json('output.json')
            with st.spinner("Integrating LLM for generating final workflow..."):
                final_data,input_token,output_token = extract_final_output(json_data)

            st.success("Data retrieved successfully!")
            st.json(final_data)

            # Print the logging stats
            # st.write("LLM 1st Call Statistics:", llm_retrieval.get_stats())
            st.write("LLM 2nd Call Statistics:","Input Tokens :",input_token,"Output Tokens :",output_token)
        else:
            st.error("Please enter a query")

if __name__ == "__main__":
    main()
