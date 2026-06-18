import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Force dotenv to override any cached terminal variables
load_dotenv(override=True)

# THE FIX: Ensure Google gets the correct Gemini API Key, not the Groq key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def generate_local_vector_db():
    source_path = "knowledge_base/ipl_2026_facts.txt"
    
    print("🟢 Step 1: Ingesting raw qualitative text data...")
    with open(source_path, "r", encoding="utf-8") as file:
        raw_data = file.read()

    print("🟢 Step 2: Segmenting text chunks via Recursive Splitter...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    text_chunks = splitter.split_text(raw_data)

    print("🟢 Step 3: Compiling math matrices via Google Text Embeddings...")
    
    # Ensure it uses the modern embedding model
    embedding_engine = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    print("🟢 Step 4: Storing chunks to local database index...")
    Chroma.from_texts(
        texts=text_chunks,
        embedding=embedding_engine,
        persist_directory="./chroma_db"
    )
    print("✅ Sprint 2.2 Complete: Knowledge Base vectorized in './chroma_db/' folder.")

if __name__ == "__main__":
    generate_local_vector_db()