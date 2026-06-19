import requests
import streamlit as st

# Generic CricAPI Endpoint (Free Tier)
API_URL = "https://api.cricapi.com/v1/currentMatches"

@st.cache_data(ttl=300) # Cache for 5 minutes (300s) to prevent CricAPI free-tier rate limit exhaustion
def fetch_live_match_data(api_key: str) -> dict:
    """
    Fetches live match data from CricAPI.
    Uses Streamlit's caching mechanism to avoid hitting rate limits.
    """
    if not api_key:
        return {"error": "API key is missing. Falling back to manual mode."}
    
    try:
        response = requests.get(f"{API_URL}?apikey={api_key}&offset=0", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "success":
            return {"error": data.get("reason", "Unknown API Error")}
            
        return data
        
    except requests.exceptions.RequestException as e:
        return {"error": f"API Request failed: {e}"}

def parse_live_match(api_data: dict) -> list:
    """
    Parses the raw API response and extracts relevant T20/ODI live matches.
    """
    if "error" in api_data:
        return []
        
    matches = api_data.get("data", [])
    parsed_matches = []
    
    for match in matches:
        match_name = match.get("name", "").lower()
        # Filter for IPL matches
        if "ipl" not in match_name and "indian premier" not in match_name:
            continue
            
        # Safely extract the score list
        score_list = match.get("score")
        
        # Check if match has started and has score data
        if match.get("matchStarted") and isinstance(score_list, list) and len(score_list) > 0:
            current_inning = score_list[-1]
            overs = current_inning.get("o", 0.0)
            runs = current_inning.get("r", 0)
            wickets = current_inning.get("w", 0)
        else:
            # For upcoming matches that haven't started
            overs = 0.0
            runs = 0
            wickets = 0
            
        parsed_matches.append({
            "name": match.get("name", "Unknown Match"),
            "venue": match.get("venue", "Unknown Venue"),
            "current_score": runs,
            "wickets": wickets,
            "balls_bowled": overs,
            "status": match.get("status", "Upcoming")
        })
            
    return parsed_matches
