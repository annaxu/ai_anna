import streamlit as st
import os
import requests
import urllib.parse
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from openai import AzureOpenAI
from langchain.tools import tool
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain import hub


split_doc_size = 1000
chunk_overlap = 50
pdf_file_name = 'data/IRM_Help.pdf'
work_dir = '/Users/I069899/Documents/study/AI/ai_anna/'
db_path =  "data/vectordb/"

env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
global llm
global new_db



@tool
def search_by_online_sap_help_docs(query_content:str):

    """online search method: search from online.

    Args:
        query_content: content used to search
    """
    
    encoded_query = urllib.parse.quote_plus(query_content)
    url = f"https://help.sap.com/http.svc/elasticsearch?area=content&version=&language=en-US&state=PRODUCTION&q={encoded_query}&transtype=standard,html,pdf,others&product=&to=19&advancedSearch=0&excludeNotSearchable=1"
    print("*222222  in search_by_online_sap_help_docs")
    response = requests.get(url)
    if response.status_code == 200:
        responseJson = response.json()
        if responseJson.get('status') == 'OK':
            #data = response.get('data', {})
            data = responseJson.get('data', {})
            results = data.get('results', [])
            for i, result in enumerate(results, 1):
                snippet_soup = BeautifulSoup(result.get('snippet', 'N/A'), 'html.parser')
                searchResult = f"Result {i}:\nTitle: {result.get('title', 'N/A')}\nDate: {result.get('date', 'N/A')}\nProduct: {result.get('product', 'N/A')}\nURL: https://help.sap.com{result.get('url', 'N/A')}\nSnippet: {snippet_soup.get_text()}\n"
                print("*222222  in search_by_online_sap_help_docs, search_result is ",searchResult)
                return searchResult
        else:
            searchResult = "No results found by query from the help.sap.com"
            print("*222222  in search_by_online_sap_help_docs, search_result is ",searchResult)
            return searchResult
    else:
        searchResult = "Request failed with status code {response.status_code}"
        print("*222222  in search_by_online_sap_help_docs, search_result is ",searchResult)
        return searchResult

def load_pdf_splitter():
    loader = PyPDFLoader(os.path.join(work_dir, pdf_file_name))
    pages = loader.load()
    text_splitter = CharacterTextSplitter(separator ="\n",chunk_size=1000,chunk_overlap=150)
    docs = text_splitter.split_documents(pages)
    return docs


@st.cache_resource
def initialize_data():
    split_docs = load_pdf_splitter()
    db = FAISS.from_documents(split_docs, AzureOpenAIEmbeddings())
    db.save_local(db_path)
    

#@st.cache_resource
@tool
def search_by_vectorDb(query_content:str):
    """Default search method: search from vector DB.

    Args:
        query_content: content used to search
    """
    # loader = PyPDFLoader(os.path.join(work_dir, pdf_file_name))
    # pages = loader.load()
    # text_splitter = CharacterTextSplitter(separator ="\n",chunk_size=1000,chunk_overlap=150)
    # split_docs = text_splitter.split_documents(pages)
    # db = FAISS.from_documents(split_docs, AzureOpenAIEmbeddings())
    # db.save_local(db_path)

    
    #llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)
    new_db = FAISS.load_local(db_path, AzureOpenAIEmbeddings())
    
    global AMAZON_REVIEW_BOT    
    AMAZON_REVIEW_BOT = RetrievalQA.from_chain_type(llm,
                retriever=new_db.as_retriever(search_type="similarity_score_threshold",
                    search_kwargs={"score_threshold": 0.75}))
                    #search_kwargs={"score_threshold": 0.5}))
    AMAZON_REVIEW_BOT.return_source_documents = True
    print("*11111111  in search_by_vectorDb, before invoke")
    ans = AMAZON_REVIEW_BOT.invoke({"query": query_content})
    #if ans["source_documents"] or enable_chat:
    if ans["source_documents"]:
        returnResult = ans["result"]
        print("11111111 in search_by_vectorDb, returnResult is ",returnResult)
        return returnResult
    else:
        returnResult = "I don't know."
        print("11111111 in search_by_vectorDb, returnResult is ",returnResult)
        return returnResult


def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
   
    initialize_data()
   
    tools = [
        Tool(
            name = "search_by_vectorDb",
            func = search_by_vectorDb.run,
            description = "default search engine. search from local vector DB. if can not find proper result, swith to online search"
        ),
        Tool(
            name = "search_by_online_sap_help_docs",
            func = search_by_online_sap_help_docs.run,
            description = "if there are no proper result by default search, use this function for online search."
        )
    ]

    prompt = hub.pull("hwchase17/openai-tools-agent")
    print("%%%%%%%% prompt is " , prompt)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False, handle_parsing_errors=True
    )

    searchResult = agent_executor.invoke({"input": message})
    return searchResult


if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
    load_dotenv(dotenv_path=env_path, verbose=True) 

    st.title('IRM Help Document')
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)

    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if prompt:
       with st.spinner("Generating......"):
           #output = chat(prompt, st.session_state["chat_history"])
           output = chat(prompt, st.session_state["chat_history"])

           st.session_state["chat_answers_history"].append(output)
           st.session_state["user_prompt_history"].append(prompt)
           st.session_state["chat_history"].append((prompt,output))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)
