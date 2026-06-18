import os
from dotenv import load_dotenv

# Import your Text Brain (RAG Pipeline)
from rag_pipeline import execute_fault_tolerant_rag

# In the future, you will import your actual Phase 1 XGBoost model here.
# For now, we will mock the ML execution call to test the fusion logic.
# from legacy_ml_pipeline import predict_score_from_xgb  

load_dotenv(override=True)

def mock_predict_score_from_xgb(live_data: dict) -> str:
    """A placeholder for your Phase 1 Math Brain."""
    # Imagine your XGBoost model calculates this based on the live_data dict
    return f"178.5 runs (Calculated via XGBRegressor based on {live_data['venue']})"

def classify_and_route(user_prompt: str, live_data: dict = None) -> str:
    """
    The Central Nervous System. Classifies intent and routes traffic 
    between the Quantitative XGBoost model and the Qualitative RAG Engine.
    """
    prompt_lower = user_prompt.lower()
    
    # 1. Intent Classification Keywords
    predictive_keywords = ["predict", "score", "will win", "chase", "target", "overs"]
    qualitative_keywords = ["who", "won", "player", "tournament", "pitch", "weather"]
    
    is_predictive = any(w in prompt_lower for w in predictive_keywords)
    is_qualitative = any(w in prompt_lower for w in qualitative_keywords)
    
    print(f"\n⚡ Orchestrator Analysis -> Predictive Intent: {is_predictive} | Qualitative Intent: {is_qualitative}")

    # ROUTE A: Hybrid Agentic Fusion (Needs Both)
    if is_predictive and is_qualitative:
        print("🔗 Fusion Route Activated: Injecting Math into Text Brain...")
        
        # 1. Get exact numbers from Math Brain
        if live_data:
            ml_prediction = mock_predict_score_from_xgb(live_data)
        else:
            ml_prediction = "[Unknown - requires live match dictionary]"
            
        # 2. PROMPT INJECTION: We secretly rewrite the user's prompt before giving it to the LLM
# 2. PROMPT INJECTION: Forcing the "Expert Analyst" persona
        agentic_prompt = (
            f"User Question: {user_prompt}\n\n"
            f"--- SYSTEM INSTRUCTION FOR LLM ---\n"
            f"You are the cricRAG Expert Analyst. You must synthesize a single, cohesive paragraph to answer the user.\n"
            f"First, confidently state that 'Based on our internal models, the projected score is {ml_prediction}.'\n"
            f"Then, smoothly transition into answering the historical part of the user's question using the provided ChromaDB context.\n"
            f"NEVER apologize. NEVER say 'I don't have information'. Write it naturally, exactly like a professional sports broadcaster."
        )
                
        # 3. Send the injected prompt to the RAG pipeline
        synthesized_response = execute_fault_tolerant_rag(agentic_prompt)
        
        return f"[🤖 FUSION SYSTEM RESPONSE]\n{synthesized_response}"

    # ROUTE B: Pure Quantitative (XGBoost Only)
    elif is_predictive:
        print("🧮 Math Brain Route Activated: Directing query straight to XGBoost...")
        if live_data:
            return f"[Math Brain Output]: Predicted Score is {mock_predict_score_from_xgb(live_data)}"
        return "[Math Brain Output]: Please provide live match parameters to generate an exact calculation."

    # ROUTE C: Pure Qualitative (ChromaDB + LLM RAG Only)
    else:
        print("🧠 Text Brain Route Activated: Directing query straight to Vector Space...")
        return execute_fault_tolerant_rag(user_prompt)


if __name__ == "__main__":
    print("-" * 60)
    print("🎛️ cricRAG Orchestrator Router Test Environment")
    print("-" * 60)
    
    # Mock live game state data for an IPL match
    mock_live_state = {
        "current_score": 120,
        "wickets": 3,
        "balls_bowled": 84, # 14 Overs
        "venue": "M Chinnaswamy Stadium"
    }
    
    # Test a Hybrid Query (Asks for a score prediction AND a historical fact)
    test_query = "If RCB is at 14 overs, predict their final score, and also remind me who won the player of the tournament in the 2026 IPL?"
    
    output = classify_and_route(test_query, live_data=mock_live_state)
    print(f"\n--- Final Orchestrated Output ---\n{output}")