import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Force dotenv to override any cached terminal variables
load_dotenv(override=True)

# THE FIX: Ensure Google gets the correct Gemini API Key, not the Groq key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Use absolute path based on project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(ROOT_DIR, "chroma_db")
KNOWLEDGE_BASE_PATH = os.path.join(ROOT_DIR, "knowledge_base", "ipl_2026_facts.txt")

def generate_local_vector_db():
    print(f"🟢 Step 1: Ingesting raw qualitative text data from {KNOWLEDGE_BASE_PATH}...")
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
            raw_data = file.read()
    except FileNotFoundError:
        print(f"❌ Error: Could not find the text file at {KNOWLEDGE_BASE_PATH}")
        return

    print("🟢 Step 2: Segmenting text chunks via Recursive Splitter...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    text_chunks = splitter.split_text(raw_data)

    print("🟢 Step 3: Compiling math matrices via Google Text Embeddings...")
    
    # Ensure it uses the modern embedding model
    embedding_engine = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    print(f"🟢 Step 4: Storing chunks to local database index at {CHROMA_PATH}...")
    Chroma.from_texts(
        texts=text_chunks,
        embedding=embedding_engine,
        persist_directory=CHROMA_PATH
    )
    print("✅ Sprint 2.2 Complete: Knowledge Base vectorized and stored.")

if __name__ == "__main__":
    generate_local_vector_db()