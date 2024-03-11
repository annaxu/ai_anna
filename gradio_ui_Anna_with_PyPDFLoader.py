import gradio as gr

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader

max_token = 8000
split_doc_size = 1000
chunk_overlap = 50
pdf_file_name = 'data/IRM_Help.pdf'
work_dir = '/Users/I069899/Documents/study/AI/ai_anna/'
db_path =  "data/vectordb/"

env_path = os.getenv("HOME") + "/Documents/src/openai/.env"


def load_pdf_splitter():
  loader = PyPDFLoader(os.path.join(work_dir, pdf_file_name))
  pages = loader.load()
  text_splitter = CharacterTextSplitter(separator ="\n",chunk_size=1000,chunk_overlap=150)
  docs = text_splitter.split_documents(pages)
  return docs

def initialize_data():
    split_docs = load_pdf_splitter()
    db = FAISS.from_documents(split_docs, AzureOpenAIEmbeddings())
    db.save_local(db_path)

    new_db = FAISS.load_local(db_path, AzureOpenAIEmbeddings())
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)
    
    global AMAZON_REVIEW_BOT    
    AMAZON_REVIEW_BOT = RetrievalQA.from_chain_type(llm,
                  #retriever=db.as_retriever(search_type="similarity_score_threshold",
                retriever=new_db.as_retriever(search_type="similarity_score_threshold",
                    search_kwargs={"score_threshold": 0.5}))
    AMAZON_REVIEW_BOT.return_source_documents = True
    return AMAZON_REVIEW_BOT


def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    ans = AMAZON_REVIEW_BOT({"query": message})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        return "I don't know."
    

def launch_ui():
    demo = gr.ChatInterface(
        fn=chat,
        title="Amazon Food Review",
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
    load_dotenv(dotenv_path=env_path, verbose=True)
    
    initialize_data()
    launch_ui()
