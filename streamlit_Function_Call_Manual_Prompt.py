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
import json

from pprint import pprint
import pandas as pd


split_doc_size = 1000
chunk_overlap = 50
pdf_file_name = 'data/IRM_Help.pdf'
work_dir = '/Users/I069899/Documents/study/AI/ai_anna/'
db_path =  "data/vectordb/"

env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
global llm
global new_db
global client



#@tool
def online_search(query_content:str):

    """online search method: search from online.

    Args:
        query_content: content used to search
    """
    
    encoded_query = urllib.parse.quote_plus(query_content)
    url = f"https://help.sap.com/http.svc/elasticsearch?area=content&version=&language=en-US&state=PRODUCTION&q={encoded_query}&transtype=standard,html,pdf,others&product=&to=19&advancedSearch=0&excludeNotSearchable=1"
    print("*222222  in online_search")
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
                print("*222222  in online_search, search_result is ",searchResult)
                return searchResult
        else:
            searchResult = "No results found by query from the help.sap.com"
            print("*222222  in online_search, search_result is ",searchResult)
            return searchResult
    else:
        searchResult = "Request failed with status code {response.status_code}"
        print("*222222  in online_search, search_result is ",searchResult)
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
#@tool
def default_search(query_content:str):
    """Default search method: search from local vector DB.

    Args:
        query_content: content used to search
    """
   
    new_db = FAISS.load_local(db_path, AzureOpenAIEmbeddings())
    
    global AMAZON_REVIEW_BOT    
    AMAZON_REVIEW_BOT = RetrievalQA.from_chain_type(llm,
                retriever=new_db.as_retriever(search_type="similarity_score_threshold",
                    #search_kwargs={"score_threshold": 0.75}))
                    search_kwargs={"score_threshold": 0.5}))
    AMAZON_REVIEW_BOT.return_source_documents = True
    print("*11111111  in default_search, before invoke")
    ans = AMAZON_REVIEW_BOT.invoke({"query": query_content})
    #if ans["source_documents"] or enable_chat:
    if ans["source_documents"]:
        returnResult = ans["result"]
        returnResult = check_result(returnResult)
        print("11111111 in default_search, returnResult is ",returnResult)
        return returnResult
    else:
        returnResult = "no_result_found"
        print("00000000 in default_search, returnResult is ",returnResult)
        return returnResult


def check_result(result):
    if "I'm sorry, I don't have enough information" in result:
        return "no_result_found"
    elif "I don't have" in result:
        return "no_result_found"
    else:
        return result
    
def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
   
    initialize_data()

    # Define Tool List for search engine.
    default_tool = {
        "type": "function", 
         "function": {
            "name": "default_search", 
            "description": "Default search engine.",
            "parameters": {
                "type":"object",
                "properties": {
                    "query_content": {
                        "type": "string", 
                        "description": "content used to search"
                    }
                }
            }
        }
    }

    online_tool = {
        "type": "function", 
         "function": {
            "name": "online_search", 
            "description": "Online Search engine.",
            "parameters": {
                "type":"object",
                "properties": {
                    "query_content": {
                        "type": "string", 
                        "description": "content used to search"
                    }
                }
            }
        }
    }

   
    #system_prompt = "You are a helpful expert. Help to search by function call default_search or online_search. If the default_search result is not no_result_found, stop call the online_search and give the answer by using the return output of default_search;   then call the online_tool with function name of  online_search.  If the default_search result is no_result_found, call online_tool ( function name is : oneline_search) and also use the return output from the function call directly as the answer content. do not use the answer by yourslef.  Do not get answer from other resources. Do not get the anwser by yourself. Do not add anything into user prompt "
    #system_prompt = "You are a helpful expert. Help to search by function call default_search and online_search. compare the two results and pick one of the answer. if the return result from default_tool( function name: default_search) is not no_result_found. then choose the default_search as the answer. otherwise choose the online_search as the answer. which is more accurate to show. Make sure the answer is using the return output message directly. and the format of the return message should also exactly the same with the return of the function call.  Do not show both of the message. Thanks. "
    system_prompt = "1. You are a helpful expert. Help to search by function call default_search and online_search. 2. Compare the two results and pick one of the answer. If the return result from default_search is NOT no_result_found, then choose the default_search as the answer and then append a note message to tell us the answer is from the default search. 3. If the return result from default_search is no_result_found then choose the online_search return result as the answer and append a note at the end of the message to tell us the answer is from online search. 4.  Make sure the answer is using the return output message directly. 5. Please make sure the format of the return message should exactly the same with the return of the function call.  6. Do not show both of the message. 7. If can not find proper result by both function tools ( default_search and online_search), try to answer by yourself and please append a note message at the end of the answer to tell us the answer is answered by chatGPT. "
    messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
            ]

    response = client.chat.completions.create(
        model="gpt-35-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tools=[default_tool,online_tool], 
        tool_choice = 'auto'
    )
    response_message = response.choices[0].message
    searchResult = "I don't know"

    if response.choices[0].finish_reason == "tool_calls":
        print("GPT asked us to call a function.")
        messages.append(response_message)

        for tool_call in response.choices[0].message.tool_calls: 
            function_name = tool_call.function.name
            params = json.loads(tool_call.function.arguments)

            if function_name == "default_search":
                print("aaaaaaa")
                function_response = default_search (
                    **params
                )
                print("aaaa function_response is " , function_response)
            else:
                print("bbbbbb")
                function_response = online_search (
                    **params
                )
                print("bbbb function_response is " , function_response)

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": function_response})

        second_response = client.chat.completions.create(
            model="gpt-35-turbo", 
            messages = messages,
        )
        searchResult = second_response.choices[0].message.content
        print("########## searchResult is : " ,searchResult)
    return searchResult


if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    #os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://pvg-azure-openai-uk-south.openai.azure.com"
    env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
    load_dotenv(dotenv_path=env_path, verbose=True) 
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2023-05-15"
    )

    st.title('SAP Document Search Engine')
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)

    user_prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if user_prompt:
       with st.spinner("Generating......"):
           output = chat(user_prompt, st.session_state["chat_history"])

           st.session_state["chat_answers_history"].append(output)
           st.session_state["user_prompt_history"].append(user_prompt)
           st.session_state["chat_history"].append((user_prompt,output))

    # Displaying the chat history
    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)
