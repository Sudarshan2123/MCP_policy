from pydantic import BaseModel, Field
from typing import Optional


class BaseConfig(BaseModel):
    RAG_MODEL   : str
    EMBEDD_MODEL: str
    CHROMA_HOST : str
    CHROMA_PORT : int
    COLLECTION_NAME : str
    MONGODB_URI: str
    DB_NAME: str
    HISTORY_COLLECTION_NAME: str
    COLLECTION_USER: str
     # --- PostgreSQL ---
    POSTGRES_HOST:     str
    POSTGRES_PORT:     int
    POSTGRES_DB:       str
    POSTGRES_USER:     str
    POSTGRES_PASSWORD: str
    # --- Redis ---
    REDIS_HOST:     str
    REDIS_PORT:     int
    REDIS_DB:       int
    REDIS_USERNAME: str
    REDIS_PASSWORD: str
    CACHE_TTL:      Optional[int]

class AnalysisQueryModel(BaseModel):
    analysis: str = Field(description="analysis of the query")
    query: str = Field(description="query")
