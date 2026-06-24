import sqlite3
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

DB_NAME = "ecommerce.db"
CHROMA_PATH = "chroma_db"

@tool
def query_sql_database(query: str) -> str:
    """Executes a SQL query against the historical_sales table to get past performance data.
    Table Schema: order_date (TEXT), sales (REAL), profit (REAL), postal_code (TEXT).
    Always return the exact numeric results."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        return f"Error executing SQL: {e}"

@tool
def query_ml_predictions(query: str) -> str:
    """Executes a SQL query against the future_predictions table to get AI forecasts.
    Table Schema: forecast_date (TEXT), predicted_sales (REAL).
    Use this when the user asks about the future, projections, or next week/month."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        return f"Error executing SQL: {e}"

@tool
def search_business_reports(query: str) -> str:
    """Searches unstructured PDF market reports (ChromaDB vector database).
    Use this to answer qualitative 'why' questions, market trends, or competitor analysis."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
        
        results = db.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        return f"Error searching vector database: {e}"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

tools = [query_sql_database, query_ml_predictions, search_business_reports]

system_prompt = (
    "You are the Autonomous Chief Operating Officer (COO) for an e-commerce company. "
    "Use your tools (query_sql_database, query_ml_predictions, search_business_reports) to answer the user's questions about sales and reports. "
    "If the question is a simple greeting or doesn't need data from the tools, respond directly."
)

coo_agent = create_react_agent(llm, tools, prompt=system_prompt)

def ask_coo(user_message: str):
    """Helper function to pass a message to the LangGraph agent and extract the response."""
    response = coo_agent.invoke({"messages": [("user", user_message)]})
    
    final_answer = response["messages"][-1].content
    return final_answer

if __name__ == "__main__":
    print("--- Testing the Autonomous COO ---")
    test_question = "What is the total sum of historical sales?"
    print(f"CEO: {test_question}")
    print(f"COO: {ask_coo(test_question)}")