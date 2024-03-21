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
    
    # encoded_query = urllib.parse.quote_plus(query_content)
    # url = f"https://help.sap.com/http.svc/elasticsearch?area=content&version=&language=en-US&state=PRODUCTION&q={encoded_query}&transtype=standard,html,pdf,others&product=&to=19&advancedSearch=0&excludeNotSearchable=1"
    # response = requests.get(url)
    # if response.status_code == 200:
    #     responseJson = response.json()
    #     if responseJson.get('status') == 'OK':
    #         #data = response.get('data', {})
    #         data = responseJson.get('data', {})
    #         results = data.get('results', [])
    #         for i, result in enumerate(results, 1):
    #             snippet_soup = BeautifulSoup(result.get('snippet', 'N/A'), 'html.parser')
    #             searchResult = f"Result {i}:\nTitle: {result.get('title', 'N/A')}\nDate: {result.get('date', 'N/A')}\nProduct: {result.get('product', 'N/A')}\nURL: https://help.sap.com{result.get('url', 'N/A')}\nSnippet: {snippet_soup.get_text()}\n"
    #             return searchResult
    #     else:
    #         searchResult = "No results found by query from the help.sap.com"
    #         return searchResult
    # else:
    #     searchResult = "Request failed with status code {response.status_code}"
    #     return searchResult
    query_content = urllib.parse.quote_plus(query_content)
    searchResult333 = f"Result :\nTitle: "'my_title'"\nDate: 'my_date_data'\nProduct: 'my_product'\nURL:'my_url'\nSnippet:'my_snippet'\n"
   
    url1 = f"https://help.sap.com/http.svc/elasticsearch?area=content&version=&language=en-US&state=PRODUCTION&q={query_content}&transtype=standard,html,pdf,others&product=&to=19&advancedSearch=0&excludeNotSearchable=1"
    url0 = f"https://help.sap.com/http.svc/elasticsearch?area=topproducts&q={query_content}&language=&state=PRODUCTION&transtype=standard,html,pdf,others&product=&to=5&excludeNotSearchable=0"
    response = requests.get(url0)
    if response.status_code == 200:
        responseJson = response.json()
        if responseJson.get('status') == 'OK':
            data = responseJson.get('data', {})
            products = data.get('products', [])
            result_strings = []
            if len(products) != 0:
                result_strings = []
                n=0
                for product in products:
                    n = n+1
                    columns = []
                    columns.append("Result: " + str(n) + "\r\n")
                    columns.append("Title: " + product.get('title', 'N/A') + "\n")
                    columns.append("Product: " +  product.get('product', 'N/A')+ "\n")
                    columns.append("URL: " + "https://help.sap.com" + product.get('url', 'N/A')+ "\n")
                    # Convert the list into a string before appending
                    result_string = ''.join(columns)
                    result_strings.append(result_string)
                final_string = '\n'.join(result_strings)
                print(final_string)
                return final_string
            else:
                response = requests.get(url1)
                if response.status_code == 200:
                    responseJson = response.json()
                    if responseJson.get('status') == 'OK':
                        data = responseJson.get('data', {})
                        results = data.get('results', [])
                        result_strings = []
                        n=0
                        if len(results) != 0:
                            for result in results:
                                n = n+1
                                columns = []
                                columns.append("Result: " + str(n) + "\r\n")
                                #result_string = "title: " + result["title"] + "\r\n"
                                columns.append("title: " + result.get('title') + "\r\n")
                                if(result["product"]):
                                    #proudct = "product: " + result.get('product', 'N/A')
                                    columns.append(result["product"] + "\r\n")
                                    #result_string +=proudct + "\r\n"
                                if(result["url"]):
                                        #columns.append(result["url"] + "\r\n")
                                        columns.append("url: " + "https://help.sap.com" +  result.get('url', 'N/A')+ "\r\n")
                                        #result_url = result.get('url', 'N/A')
                                        #result_string +="https://help.sap.com" + result_url + "\r\n"
                                if result["snippet"]:
                                    columns.append(str(BeautifulSoup(result.get('snippet', 'N/A'), 'html.parser')) + "\r\n")
                                    #result_string +=str(BeautifulSoup(result.get('snippet', 'N/A'), 'html.parser')) + "\r\n"
                                    #result_string += + "\n" + result["snippet"] 
                                if result["date"]:
                                   #result_string += "\n Last updated: " + result["date"]  + "\r\n"
                                    columns.append("Last updated: " + result["date"] + "\r\n")
                                result_string = ''.join(columns)
                                result_strings.append(result_string)
                            final_string = "\n".join(result_strings)
                            print(final_string)
                            return final_string
        else:
            searchResult = "No results found by query from the help.sap.com"
            return searchResult
    else:
        searchResult = f"Request failed with status code {response.status_code}"
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
    ans = AMAZON_REVIEW_BOT.invoke({"query": query_content})
    if ans["source_documents"]:
        returnResult = ans["result"]
        returnResult = check_result(returnResult)
        print(" default search returnResult is " ,returnResult)
        return returnResult
    else:
        returnResult = "no_result_found"
        return returnResult


def check_result(result):
    if "I'm sorry, I don't have enough information" in result:
        return "no_result_found"
    elif "I don't have" in result:
        return "no_result_found"
    else:
        return result
    
def chat(message, history):
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

   # system_prompt = "1. You are a helpful expert. Help to search according to user input. 2. Make sure the answer is using the return output message directly. 3. Please make sure the format of the return message should exactly the same with the return of the function call. "
    system_prompt = "1. You are a helpful expert. Help to search according to user input. If there are multiple results, Please return the top3 most relevant result."
    messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
            ]
    default_search_result = default_search(user_prompt)
    if(default_search_result!="no_result_found"):
        return default_search_result
    else:
       response = client.chat.completions.create(
        model="gpt-35-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tools=[online_tool], 
        tool_choice = 'auto'
       )
    response_message = response.choices[0].message
    if response.choices[0].finish_reason == "tool_calls":
        print("GPT asked us to call a function.")
        messages.append(response_message)

        for tool_call in response.choices[0].message.tool_calls: 
            function_name = tool_call.function.name
            params = json.loads(tool_call.function.arguments)

            if function_name == "online_search":
                function_response = online_search (
                    **params
                )
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": function_response})

        second_response = client.chat.completions.create(
            model="gpt-35-turbo", 
            messages = messages,
        )
        searchResult = second_response.choices[0].message.content
        print("########## second_response searchResult is : " ,searchResult)
                
        return searchResult
    else:
        print("$$$$$$$$$$ finish_reason is not tool_calls. response_message is ",response_message)
        return "can not find both from local vector DB as well as oneline search. Please try another input."


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
