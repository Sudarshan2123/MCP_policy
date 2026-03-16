import os
import shutil
import sys
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # ← MCP Inspector captures this
    force=True
)

logger = logging.getLogger(__name__)

import html
import threading
import sys
import chromadb
from mcp.server.fastmcp import FastMCP
from ChatProcess import ChatProcess
from chatprocess.vector_search import mmr_search, run_sql_agent
from init import get_pipeline
import logging
from sqlalchemy import QueuePool, create_engine, text


mcp = FastMCP("Policy retrival tool", log_level="ERROR",host="0.0.0.0",port=8080)

def connect():
    try:
        pipeline = get_pipeline()

        engine_url = (
                f"postgresql+psycopg2://{pipeline.config.POSTGRES_USER}:{pipeline.config.POSTGRES_PASSWORD}@"
                f"{pipeline.config.POSTGRES_HOST}:{pipeline.config.POSTGRES_PORT}/{pipeline.config.POSTGRES_DB}"
            )
        pipeline.engine = create_engine(
            engine_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_timeout=30,
            echo_pool=False
            )

        query = ("""  SELECT c.table_name, array_agg(c.column_name ORDER BY c.ordinal_position) AS columns
    FROM information_schema.columns c
    JOIN information_schema.tables t
        ON c.table_name = t.table_name AND c.table_schema = t.table_schema
    WHERE c.table_schema = 'HRMS_QA' AND t.table_type = 'BASE TABLE'
    GROUP BY c.table_name
                 """
            )

        with pipeline.engine.connect() as conn:
            result = conn.execute(text(query))
            table_names = [(row[0].lower(),row[1]) for row in result.fetchall()]

        logger.info(f"Retrieved {len(table_names)} tables from DB")

        chroma_path = "./chroma_db"
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)

        chroma_client = chromadb.PersistentClient(path=chroma_path)
        pipeline.chroma_collection = chroma_client.create_collection(
        name="table_collection",
        metadata={"hnsw:space": "cosine"}
        )
        collection = pipeline.chroma_collection  # ✅ local alias for use below
        logger.info("ChromaDB collection created")


        for (table_name,columns) in table_names:
            if isinstance(columns, str):
                columns = columns.strip('{}').split(',')

            col_string = ', '.join(col.strip() for col in columns)

            doc_text = (
                f"Table Name: {table_name}. "
                f"Columns: {col_string}. "
            )
            logger.info(f"Processing table: {table_name}, Columns: {col_string}")

            embedding = pipeline.embeddings.embed_query(doc_text)
            logger.info(f"Generated embedding for table {table_name} with dimension: {len(embedding)}")
            collection.upsert(
                ids=[table_name],
                embeddings=[embedding],
                metadatas= [{"table_name": table_name,"columns":col_string}],
                documents=[doc_text],
            )

        logger.info("All tables processed and upserted into ChromaDB successfully")

    except Exception as e:
        return(f"There is issue in the MCP server for the retrival of the Information")


@mcp.tool(
    name = 'Policy_RAG_Implementation',
    description= 'The following tool is used for retrival of the policy documents for the chroma db according to the user provided question',
)
async def policy(user_input:str) ->str:
    try:
        pipeline = get_pipeline()
        ChatProcess_obj = ChatProcess()
        with_message_history = ChatProcess_obj.Rag_chain()
        response = with_message_history.invoke({"question":user_input },config={"configurable": {"session_id": pipeline.session_id}})
        response_answer = html.escape(response)
        return response_answer
    except Exception as e:
        return(f"There is issue in the MCP server for the retrival of the Policy")

@mcp.tool(
    name = 'Structured_data_analysis',
    description= 'The following tool is used for retrival of the information from database like postgres according to the user provided question and it also returns the sources of the retrived documents',
)
def sqlagent(user_input:str) -> str:
    try:
        relevant_tables = mmr_search(user_input = user_input)
        answer = run_sql_agent(relevant_tables,user_input)
        return answer
    except Exception as e:
        return(f"There is issue in the MCP server for the retrival of the Information")

# Configure logging at the very top of your main block
if __name__ == "__main__":
    init_thread = threading.Thread(target=connect, daemon=True)
    init_thread.start()
    mcp.run(transport="sse")

