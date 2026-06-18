import streamlit as st

# Import your Phase 3 Orchestrator
from backend.orchestrator import classify_and_route

# --- Page Configuration ---
st.set_page_config(page_title="cricRAG Predictor", page_icon="🏏", layout="wide")

# --- Sidebar Controls (Live Match Data) ---
st.sidebar.header("⚙️ Live Match Parameters")
st.sidebar.markdown("Slide to update the match state in real-time.")

venue = st.sidebar.selectbox("Venue", ["M Chinnaswamy Stadium", "Wankhede Stadium", "Eden Gardens"])
current_score = st.sidebar.slider("Current Score", 0, 250, 120)
wickets = st.sidebar.slider("Wickets Down", 0, 10, 3)
overs = st.sidebar.slider("Overs Bowled", 0.0, 19.5, 14.0, step=0.1)

# Package the slider data into the dictionary the Orchestrator expects
live_match_state = {
    "venue": venue,
    "current_score": current_score,
    "wickets": wickets,
    "balls_bowled": overs
}

# --- Main Dashboard ---
st.title("🏏 cricRAG Analytical Engine")
st.markdown("### Dual-Architecture T20 Forecasting & Qualitative RAG")

col1, col2, col3 = st.columns(3)
crr = round(current_score / max(overs, 1), 2)
col1.metric("Current Run Rate", f"{crr}")
# In a real app, you might run predict_score_from_xgb here on every slider move.
# For now, we will wait for the user to ask the chat.
col2.metric("Projected Score", "Ask Chat for ML Output...") 
col3.metric("Live Phase", "Death Overs" if overs >= 16 else ("Powerplay" if overs <= 6 else "Middle Overs"))

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