# 🏏 cricRAG Analytical Engine: Full-Stack AI & ML Pipeline

An enterprise-grade, dual-architecture AI system designed to forecast T20 cricket match outcomes and provide context-aware qualitative analysis via Retrieval-Augmented Generation (RAG).

**Current Status:** Phase 1 (The Math Brain) — **Completed**. 
The core Machine Learning pipeline predicts the final first-innings score of a T20 match from any live game state with a hyper-optimized Mean Absolute Error (MAE) of **12.43 runs**.

---

## 🏗️ System Architecture

This application is built using a multi-layered generative and predictive architecture, divided into four distinct engineering phases:

* **🟢 Phase 1: The Math Brain (Completed)**
  * A Gradient Boosting machine (`XGBRegressor`) trained on 18 years of historical ball-by-ball delivery data to calculate live momentum, spatial venue constraints, and strategic game phases.
* **🟡 Phase 2: The Text Brain (Upcoming)**
  * A Retrieval-Augmented Generation (RAG) agent powered by an LLM to read qualitative data (pitch reports, weather, historical context) and answer semantic queries.
* **⚪ Phase 3: The Orchestrator (Upcoming)**
  * A semantic routing engine that dynamically passes user queries to either the Math Brain, the Text Brain, or fuses them both.
* **⚪ Phase 4: Frontend UI & Deployment (Upcoming)**
  * An interactive Streamlit dashboard for real-time inference and chat.

---

## 🚀 Phase 1: The Analytical Engine (Machine Learning Pipeline)

Real-world sports forecasting requires dealing with highly non-linear data and fractured historical records. Phase 1 focused heavily on data sanitization and advanced feature engineering before passing the data to the algorithm.

### 📊 Model Performance
* **Algorithm:** `XGBRegressor`
* **Hyperparameters:** `n_estimators=400`, `learning_rate=0.05`, `max_depth=7`, `subsample=0.8`
* **Baseline MAE:** 15.19 runs
* **Optimized MAE:** **12.43 runs**

### 🛡️ Data Sanitization & Threat Neutralization
The raw dataset required a robust sanitization pipeline to prevent mathematical skewing:
1. **The Super Over Bug (Critical):** Discovered and programmatically dropped 6-ball "Super Overs" (innings 3-6) that were poisoning the model's ability to regress a standard 120-ball score.
2. **Spatial/Venue Fractures:** Engineered a master Pandas dictionary to unify decades of stadium rebrands and naming inconsistencies (e.g., mapping *M Chinnaswamy Stadium, Bengaluru* and its variations into a single continuous baseline).
3. **Categorical Timeline Merging:** Standardized modern acronyms (RCB, CSK) with their historical full names to ensure the model views a franchise's 18-year history as an unbroken timeline.
4. **Metadata Patching:** Injected synthetic "Unknown" flags for missing qualitative data (`toss_winner`, `umpire`) to secure the pipeline for the Phase 2 RAG integration.

### ⚙️ Feature Engineering
The raw delivery logs were mathematically transformed into dynamic predictive features:
* **Spatial Intelligence:** `venue_avg_score` calculated per stadium (dropping sample sizes < 5 matches) to teach the model boundary dimensions and pitch behavior.
* **Momentum Tracking:** Engineered `crr` (Current Run Rate) utilizing `(current_score * 6) / np.maximum(balls_bowled, 1)` to prevent division-by-zero errors on the opening delivery.
* **Strategic Game Phases:** Built binary flags (`is_powerplay` for overs 1-6, `is_death` for overs 16-20) allowing the gradient booster to apply dynamic mathematical weights to runs scored against spread vs. restricted fielding setups.

---

## 🗺️ Project Roadmap: Evolution to Full-Stack AI

This project is being developed in iterative, agile sprints. With the quantitative baseline secured, the following phases will integrate Generative AI and build the user-facing application.

### 🟡 Phase 2: The Text Brain (RAG Architecture)
*Goal: Build an AI agent capable of reading qualitative cricket data (pitch reports, weather, historical context) that the quantitative XGBoost model cannot process.*

* **Sprint 2.1 - LLM Integration:** Connect the Python backend to a Large Language Model (e.g., Google Gemini or OpenAI) via API, establishing a base system prompt to act as an expert cricket analyst.
* **Sprint 2.2 - Knowledge Base (Vector Database):** Compile missing metadata and text trivia. Utilize an embedding model to convert text chunks into high-dimensional vectors and store them in a local Vector DB (e.g., ChromaDB or FAISS).
* **Sprint 2.3 - Retrieval Pipeline:** Build the logic that takes a user's qualitative query (e.g., "Was the pitch spinning in the second innings?"), performs a semantic similarity search in the Vector DB, and injects the retrieved context into the LLM's prompt for a hallucination-free response.

### ⚪ Phase 3: The Orchestrator (The Fusion Engine)
*Goal: Create a semantic router acting as the central nervous system, intelligently directing traffic between the Math Brain and the Text Brain.*

* **Sprint 3.1 - Intent Classification:** Build a routing script (`orchestrator.py`) that analyzes incoming user prompts. 
  * If the prompt asks for "score," "target," or "prediction" → Route to XGBoost (`Math Brain`).
  * If the prompt asks for "toss," "weather," or "pitch" → Route to the RAG Pipeline (`Text Brain`).
* **Sprint 3.2 - Response Synthesis:** Write fusion logic for dual-intent queries. If a user asks for *both* a prediction and pitch context, the Orchestrator will run both pipelines simultaneously and have the LLM format the mathematical output and retrieved text into one cohesive response.

### ⚪ Phase 4: Frontend UI & Deployment
*Goal: Wrap the dual-brain architecture in a sleek, interactive web application.*

* **Sprint 4.1 - Streamlit Architecture:** Build a pure Python web dashboard using `streamlit`.
  * **Sidebar:** Interactive sliders and dropdowns (Inning, Venue, Current Score, Balls Left) that feed directly into the XGBoost inference API.
  * **Main Chat Interface:** A native chat window where users can talk to the RAG agent and query historical trivia.
* **Sprint 4.2 - State Management & Deployment:** Implement `st.session_state` to maintain conversational memory. Containerize the application and deploy it live to the internet for public portfolio access.

---

## 🛠️ Tech Stack & Engineering Tools

* **Core Language:** Python 3.x (Isolated via `venv`)
* **Data Engineering:** `pandas` (complex relational joins, rolling aggregations), `numpy` (matrix constraints)
* **Machine Learning:** `scikit-learn` (pipeline validation, error metrics), `xgboost` (gradient boosting decision trees)
* **Generative AI (Upcoming):** LLM APIs (Gemini/OpenAI), `langchain`, `chromadb` (Vector Database)
* **Frontend (Upcoming):** `streamlit`
* **MLOps / Deployment:** `joblib` (serialization of the trained model and feature architecture into lightweight `.pkl` binaries for rapid API inference).

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/cricRAG-analytical-Engine.git](https://github.com/yourusername/cricRAG-analytical-Engine.git)
cd cricRAG-analytical-Engine

graph TD
    %% Define Styles
    classDef frontend fill:#ff4b4b,stroke:#fff,stroke-width:2px,color:#fff
    classDef router fill:#feca57,stroke:#333,stroke-width:2px,color:#333
    classDef mathBrain fill:#48dbfb,stroke:#333,stroke-width:2px,color:#333
    classDef textBrain fill:#1dd1a1,stroke:#333,stroke-width:2px,color:#333
    classDef database fill:#ff9f43,stroke:#333,stroke-width:2px,color:#333

    %% Nodes
    USER((👤 User))
    UI[🖥️ Streamlit Frontend]:::frontend
    
    ORCH{🧠 Orchestrator \nIntent Classification}:::router
    
    MATH[🧮 Phase 1: XGBoost \nQuantitative Engine]:::mathBrain
    
    TEXT_PIPELINE[📚 Phase 2: RAG Pipeline]:::textBrain
    VDB[(🗄️ ChromaDB \nVector Space)]:::database
    LLM_ROUTER{⚙️ Fault-Tolerant \nLLM Router}:::router
    GEMINI[🔹 Google Gemini \nPrimary]:::textBrain
    GROQ[🔸 Groq Llama 3.1 \nFallback]:::textBrain
    
    FINAL[💬 Synthesized \nBroadcaster Output]:::frontend

    %% Connections
    USER -->|Adjusts Live Sliders| UI
    USER -->|Types Text Prompt| UI
    
    UI -->|Match State + Query| ORCH
    
    %% Routing Paths
    ORCH -->|Route B: Pure Math| MATH
    ORCH -->|Route C: Pure Text| TEXT_PIPELINE
    
    %% The Hybrid Magic
    ORCH -->|Route A: Hybrid Fusion| MATH
    MATH -->|Calculated Score Injected \ninto System Prompt| TEXT_PIPELINE
    
    %% RAG Internal Flow
    TEXT_PIPELINE -->|Retrieves Ground Truth| VDB
    VDB -->|Historical Context| LLM_ROUTER
    
    LLM_ROUTER -->|Attempt 1| GEMINI
    LLM_ROUTER -->|If API Fails / Rate Limit| GROQ
    
    %% Output
    GEMINI --> FINAL
    GROQ --> FINAL
    MATH -.->|If Route B| FINAL
    FINAL --> UI
