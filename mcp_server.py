import html
from mcp.server.fastmcp import FastMCP
from ChatProcess import ChatProcess
from init import get_pipeline

mcp = FastMCP("Policy retrival tool", log_level="INFO")

@mcp.tool(
    name = 'Policy_RAG_Implementation',
    description= 'The following tool is used for retrival of the policy documents for the chroma db according to the user provided question',
)
def policy(user_input:str) ->str:
    try:
        pipeline = get_pipeline()
        ChatProcess_obj = ChatProcess()
        with_message_history = ChatProcess_obj.Rag_chain()
        response = with_message_history.invoke({"question":user_input },config={"configurable": {"session_id": pipeline.session_id}})
        response_answer = html.escape(response)
        return response_answer
    except Exception as e:
        return(f"There is issue in the MCP server for the retrival of the Policy")

if __name__ == "__main__":
    mcp.run(transport="sse")
