
import uuid
import chromadb
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from pydantic import config
from base_config import BaseConfig
from configuration import Configuration_Manager
from gcp import load_gcp_credentials

_chroma_client: Optional[chromadb.HttpClient] = None

class Init:
    def __init__(self) -> None:
        self.config_obj = Configuration_Manager()
        self.config=self.config_obj.get_base_config()
        credentials= load_gcp_credentials()
        Chroma_host = self.config.CHROMA_HOST
        Chroma_port = self.config.CHROMA_PORT
        self.model_llm = ChatGoogleGenerativeAI(
            model = self.config.RAG_MODEL,
            temperature = 0.3,
            max_output_tokens = 1600,
            credentials = credentials
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.config.EMBEDD_MODEL,credentials=credentials)
        self._chroma_client=chromadb.HttpClient(
            host = Chroma_host,
            port = Chroma_port,
        )
        self.Collection_Name = self.config.COLLECTION_NAME
        self.session_id = str(uuid.uuid4()) 

_pipeline = Init()

def get_pipeline() -> Init:
    return _pipeline
