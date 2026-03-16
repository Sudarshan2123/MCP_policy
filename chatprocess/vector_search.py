import chromadb
from sqlalchemy import text
from init import get_pipeline
import logging
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

from sklearn.metrics.pairwise import cosine_similarity


def mmr_search(user_input: str, top_k: int = 6, lambda_val: float = 0.8, fetch_k: int = 20):

    pipeline = get_pipeline()
    # ✅ Reuse the same EphemeralClient collection built in connect()
    collection = pipeline.chroma_collection

    if collection is None:
        logger.warning("ChromaDB collection not ready yet")
        return []

    q_embedding = pipeline.embeddings.embed_query(user_input)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=fetch_k,
        include=["embeddings","metadatas","documents"]
    )

    candidate_ids = results["ids"][0]
    candidate_embeddings = results["embeddings"][0]
    candidate_docs = results["documents"][0]
    candidate_metadata = results["metadatas"][0]

    selected_ids = []
    remaining = list(range(len(candidate_ids)))

    while len(selected_ids) < top_k and remaining:
        mmr_scores = {}
        for idx in remaining:
            relevance = cosine_similarity([q_embedding], [candidate_embeddings[idx]])[0][0]
            if not selected_ids:
                diversity = 0
            else:
                diversity = max(cosine_similarity([candidate_embeddings[idx]], [candidate_embeddings[i] for i in selected_ids])[0])

            mmr_scores[idx] = lambda_val * relevance - (1 - lambda_val) * diversity

        best_idx = max(mmr_scores,key=mmr_scores.get)
        selected_ids.append(best_idx)
        remaining.remove(best_idx)


    final_results = []
    for idx in selected_ids:
        final_results.append({
            "metadata": candidate_metadata[idx]
        })

    return final_results

def run_sql_agent(relevant_tables,user_input):
    SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an expert PostgreSQL SQL generator for an HRMS database.

    Given the relevant tables and their columns, generate a single valid PostgreSQL SQL query to answer the user's question.

    Rules:
    - Use ONLY the tables and columns provided
    - CRITICAL: You must always double-quote the schema and table names.
      Example: "HRMS_QA"."employ_leave_master"
    - Always use schema prefix: "HRMS_QA".<table_name>
    - Return ONLY the raw SQL query, no markdown, no backticks, no explanation
    - If the question cannot be answered with the given tables, return exactly: INSUFFICIENT_TABLES
    """),
        ("human", """Relevant Tables:
    {table_context}

    Question: {question}

    SQL Query:""")
    ])


    ANSWER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an HRMS data analyst assistant.
    Given the SQL query results, answer the user's question in clear, concise natural language.
    - Be specific with numbers and names from the results
    - If results are empty, say no matching data was found
    - Do NOT mention SQL or technical details unless specifically asked
    """),
        ("human", """Question: {question}

    Query Results: {results}

    Answer:""")
    ])
    try:
        pipeline = get_pipeline()
        llm = pipeline.model_llm

        relevant_tables = relevant_tables
        logger.info(f"Retrieved {len(relevant_tables)} relevant tables")

        if not relevant_tables:
            logger.warning("No relevant tables found for the given question.")
            return "No relevant tables found for the given question."

        table_context = "\n".join([
    f"- Table: {t['metadata']['table_name']} | Columns: {t['metadata']['columns']}"
    for t in relevant_tables
])
        sql_chain = SQL_GENERATION_PROMPT | llm
        sql_response = sql_chain.invoke({
            "table_context": table_context,
            "question": user_input
        })

        generated_sql = sql_response.content.strip()
        logger.info(f"Generated SQL: {generated_sql}")

        with pipeline.engine.connect() as conn:
                    result = conn.execute(text(generated_sql))
                    columns = list(result.keys())
                    rows = result.fetchall()

        query_results = [dict(zip(columns, row)) for row in rows]
        logger.info(f"Query Result: {query_results}")

        answer_chain = ANSWER_PROMPT | llm
        answer_response = answer_chain.invoke({
            "question": user_input,
            "results": str(query_results[:50])  # cap at 50 rows in context
        })

        final_ans = answer_response.content.strip()
        logger.info(f"Final Answer: {final_ans}")
        return final_ans

    except Exception as e:
            logger.error(f"run_sql_agent failed: {e}", exc_info=True)
            return f"An error occurred while processing your question: {e}"

