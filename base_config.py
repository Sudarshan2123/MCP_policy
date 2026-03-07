from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    RAG_MODEL   : str
    EMBEDD_MODEL: str
    CHROMA_HOST : str
    CHROMA_PORT : int
    COLLECTION_NAME : str
    MONGODB_URI: str
    DB_NAME: str
    HISTORY_COLLECTION_NAME: str

class AnalysisQueryModel(BaseModel):
    analysis: str = Field(description="analysis of the query")
    query: str = Field(description="query")
