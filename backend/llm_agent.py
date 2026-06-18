import os
from google import genai
from google.genai import types
from groq import Groq
from dotenv import load_dotenv

# 1. Load Environment Configuration
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_KEY or not GROQ_KEY:
    raise ValueError("🚨 Missing API keys in .env file! Check GEMINI_API_KEY and GROQ_API_KEY.")

# 2. Initialize Modern Clients
google_client = genai.Client(api_key=GEMINI_KEY)
groq_client = Groq(api_key=GROQ_KEY)

# 3. System Personality Guardrails
SYSTEM_PROMPT = """
You are an elite, professional IPL Cricket Analyst and Data Scientist. 
Provide factual, structured, and data-driven answers based on the context provided.
If a query is unrelated to cricket or your analytical tools, politely decline to answer.
Never hallucinate or invent statistics.
"""

def query_primary_model(prompt: str) -> str:
    """Invokes the primary Gemini engine using the modern SDK."""
    response = google_client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2
        )
    )
    return response.text

def query_fallback_model(prompt: str) -> str:
    """Invokes the backup Groq Llama 3.1 engine for a completely free failover."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant", # <--- THE NEW 2026 STANDARD
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content