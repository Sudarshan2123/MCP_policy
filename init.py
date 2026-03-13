
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
        self.chroma_client2 = chromadb.EphemeralClient()         # for table_collection (SQL agent)
        self.chroma_collection = None
        self.session_id = str(uuid.uuid4())
        self.engine = None
        self.collection_user = self.config.COLLECTION_USER
        self.postgres_host = self.config.POSTGRES_HOST
        self.postgres_port = self.config.POSTGRES_PORT
        self.postgres_db = self.config.POSTGRES_DB
        self.postgres_pass = self.config.POSTGRES_PASSWORD
        self.redis_host = self.config.REDIS_HOST
        self.redis_port = self.config.REDIS_PORT
        self.redis_db = self.config.REDIS_DB
        self.redis_user = self.config.REDIS_USERNAME
        self.redis_pass = self.config.REDIS_PASSWORD
        self.cache_ttl = self.config.CACHE_TTL



_pipeline = Init()

def get_pipeline() -> Init:
    return _pipeline
