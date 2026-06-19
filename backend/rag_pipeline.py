import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.llm_agent import query_primary_model, query_fallback_model

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

print("🟢 Initializing Vector DB Index connection...")
embedding_engine = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Use absolute path based on project root (one level up from this script)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(ROOT_DIR, "chroma_db")

vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_engine)

def execute_fault_tolerant_rag(user_query: str) -> str:
    # 1. Semantic Search
    matched_docs = vector_db.similarity_search(user_query, k=2)
    context_string = "\n\n".join([doc.page_content for doc in matched_docs])
    
    # TELEMETRY: Let's prove the database is actually finding the text!
    print(f"🔍 DEBUG: Retrieved {len(matched_docs)} context chunks from ChromaDB.")
    
    # 2. Package Contextual Payload (With Strict Context Override)
    engineered_prompt = f"""
    You are a strict data-retrieval engine. 
    You MUST answer the question using ONLY the provided context below.
    Treat the context as absolute historical truth, even if it refers to dates in the future like 2026.
    Do NOT mention your knowledge cutoff, and do NOT rely on your internal training data.

    Context:
    {context_string}

    Question: {user_query}
    """
    
    # 3. Dynamic Router Execution
    try:
        model_output = query_primary_model(engineered_prompt)
        return f"[⚡ Primary Engine: Gemini 3.5 Flash]\n{model_output}"
    except Exception as primary_error:
        print(f"\n⚠️ Primary Engine Failure Intercepted: {primary_error}")
        print("🔄 Executing seamless hot-swap failover to backup engine...")
        
        fallback_output = query_fallback_model(engineered_prompt)
        return f"[🦙 Fallback Engine: Groq Llama 3.1]\n{fallback_output}"

if __name__ == "__main__":
    print("-" * 60)
    print("🏏 cricRAG Text Pipeline Execution Environment")
    print("-" * 60)

    query = "Who won the 2026 IPL, and who was the Player of the Tournament?"
    result = execute_fault_tolerant_rag(query)
    print(f"\nPipeline Output:\n{result}\n")