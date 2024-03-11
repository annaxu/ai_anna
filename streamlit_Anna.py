import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


split_doc_size = 1000
chunk_overlap = 50
pdf_file_name = 'data/IRM_Help.pdf'
work_dir = '/Users/I069899/Documents/study/AI/ai_anna/'
db_path =  "data/vectordb/"

env_path = os.getenv("HOME") + "/Documents/src/openai/.env"


# def load_pdf_splitter():
#   loader = PyPDFLoader(os.path.join(work_dir, pdf_file_name))
#   pages = loader.load()
#   text_splitter = CharacterTextSplitter(separator ="\n",chunk_size=1000,chunk_overlap=150)
#   docs = text_splitter.split_documents(pages)
#   return docs

@st.cache_resource
def initialize_data():
    #split_docs = load_pdf_splitter()
   #  db = FAISS.from_documents(split_docs, AzureOpenAIEmbeddings())
   #  db.save_local(db_path)

    new_db = FAISS.load_local(db_path, AzureOpenAIEmbeddings())
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)
    
    global AMAZON_REVIEW_BOT    
    AMAZON_REVIEW_BOT = RetrievalQA.from_chain_type(llm,
                retriever=new_db.as_retriever(search_type="similarity_score_threshold",
                    search_kwargs={"score_threshold": 0.5}))
    AMAZON_REVIEW_BOT.return_source_documents = True
    return AMAZON_REVIEW_BOT




def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    AMAZON_REVIEW_BOT = initialize_data()

    ans = AMAZON_REVIEW_BOT.invoke({"query": message})
    if ans["source_documents"] or enable_chat:
        return ans["result"]
    else:
        return "I don't know."

if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
    load_dotenv(dotenv_path=env_path, verbose=True) 

    st.title('IRM Help Document')

    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if prompt:
       with st.spinner("Generating......"):
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
