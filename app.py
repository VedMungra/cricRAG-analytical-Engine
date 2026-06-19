import os
import streamlit as st
from dotenv import load_dotenv

# Import your Phase 3 Orchestrator
from backend.orchestrator import classify_and_route
from backend.api_integration import fetch_live_match_data, parse_live_match

load_dotenv()
API_KEY = os.getenv("CRIC_API_KEY", "")

# --- Page Configuration ---
st.set_page_config(page_title="cricRAG Predictor", page_icon="🏏", layout="wide")

@st.dialog("⚠️ Important Disclaimer")
def show_disclaimer():
    st.markdown("""
    **Please read before using cricRAG:**
    - This analytical engine is for **educational and informational purposes only**.
    - The ML predictions and AI chat are purely statistical and may contain inaccuracies.
    - **Do not** use this tool for betting, gambling, or any financial decisions.
    - **API Limits:** Live data automatically refreshes every 5 minutes. The free-tier API has a strict 100 hits/day limit, so please use the manual "Refresh Live Data" button sparingly to avoid hitting the cap!
    """)
    if st.button("I Understand & Agree"):
        st.session_state.disclaimer_accepted = True
        st.rerun()

if "disclaimer_accepted" not in st.session_state:
    show_disclaimer()

# --- Sidebar Controls (Live Match Data) ---
st.sidebar.header("⚙️ Match Configuration")

live_match_state = {}

if st.sidebar.button("🔄 Refresh Live Data"):
    fetch_live_match_data.clear()
    st.rerun()

if not API_KEY:
    st.sidebar.warning("⚠️ CRIC_API_KEY not found in .env. Cannot fetch live data.")
else:
    with st.sidebar.status("Fetching IPL Matches...", expanded=True):
        api_data = fetch_live_match_data(API_KEY)
        matches = parse_live_match(api_data)
        
    if "error" in api_data and api_data["error"]:
        st.sidebar.error(f"API Error: {api_data['error']}")
    elif not matches:
        st.sidebar.info("There are no upcoming IPL matches.")
    else:
        match_names = [m["name"] for m in matches]
        selected_match_name = st.sidebar.selectbox("Select IPL Match", match_names)
        selected_match = next((m for m in matches if m["name"] == selected_match_name), None)
        
        if selected_match:
            st.sidebar.success(f"Status: {selected_match['status']}")
            live_match_state = {
                "venue": selected_match["venue"],
                "current_score": selected_match["current_score"],
                "wickets": selected_match["wickets"],
                "balls_bowled": selected_match["balls_bowled"]
            }

if not live_match_state:
    # Fallback empty state if no match is available
    live_match_state = {
        "venue": "N/A",
        "current_score": 0,
        "wickets": 0,
        "balls_bowled": 0.0
    }

# --- Main Dashboard ---
st.title("🏏 cricRAG Analytical Engine")
st.markdown("### Dual-Architecture T20 Forecasting & Qualitative RAG")

with st.expander("📖 How to Use this App"):
    st.markdown("""
    **1. Select the Live Match (Sidebar)**
    - The app automatically fetches real-time data for active or upcoming IPL matches from the Live API.
    - Select your desired match from the dropdown in the sidebar to load its real-time state.

    **2. Review Current Metrics**
    - The dashboard will instantly calculate the Current Run Rate and determine the current match phase.

    **3. Ask the AI for Predictions**
    - Scroll down to the **Ask the Orchestrator** chat box.
    - Type a prompt like: *"Predict the final score for this match"* or *"What's the pitch report here?"*
    - The AI will analyze the real-time match context, run it through the Machine Learning model, and return your prediction!
    """)

col1, col2, col3 = st.columns(3)
current_score = live_match_state.get("current_score", 0)
overs_bowled = live_match_state.get("balls_bowled", 0.0)
crr = round(current_score / max(overs_bowled, 1), 2) if overs_bowled > 0 else 0.0

with col1:
    st.metric(label="📈 Current Run Rate", value=f"{crr:.2f}")

with col2:
    st.metric(label="🎯 Projected Score", value="-", help="Scroll down to the chat to get the XGBoost projected score!") 

with col3:
    phase = "Death Overs" if overs_bowled >= 16 else ("Powerplay" if overs_bowled <= 6 and overs_bowled > 0 else ("-" if overs_bowled == 0 else "Middle Overs"))
    st.metric(label="🕒 Match Phase", value=phase)

st.divider()

# --- Chat Interface (Text Brain) ---
st.subheader("💬 Ask the Orchestrator")

# 1. Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display historical messages on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Chat input loop
if user_query := st.chat_input("E.g., 'Predict the final score and tell me the pitch report.'"):
    # Display and save user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
        
    # Trigger Orchestrator, display, and save AI response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Orchestrator routing query..."):
            ai_response = classify_and_route(user_query, live_data=live_match_state)
            st.markdown(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})