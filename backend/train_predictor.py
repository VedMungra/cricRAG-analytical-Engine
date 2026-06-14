import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_math_brain():
    print("Loading data...")
    # low_memory=False to silence the DtypeWarning
    deliveries = pd.read_csv("data/deliveries.csv", low_memory=False)
    matches = pd.read_csv("data/matches.csv")
    
    # Merge datasets on the unique match identifier
    df = pd.merge(deliveries, matches, left_on='match_id', right_on='id')
    
    # ==========================================
    # DATA SANITIZATION & THREAT NEUTRALIZATION
    # ==========================================
    print("Sanitizing data and neutralizing threats...")
    
    # 🚨 CRITICAL THREAT FIX: Remove Super Overs
    # We strictly enforce standard 120-ball innings (1 and 2)
    df = df[df['inning'].isin([1, 2])]
    
    # Filter out rain-affected matches
    if 'method' in df.columns:
        df = df[df['method'] != 'D/L']

    # ⚠️ MEDIUM THREAT FIX: Standardize Dismissal Spelling
    # Ensures future analytics don't split "run out" and "runout"
    dismissal_mappings = {
        'runout': 'run out',
        'hitwicket': 'hit wicket'
    }
    df['dismissal_kind'] = df['dismissal_kind'].replace(dismissal_mappings)

    # ℹ️ LOW THREAT FIX: Patch Missing Metadata
    # Fills missing 2025/2026 data with "Unknown" so the Phase 2 RAG Agent doesn't crash on NaNs
    metadata_cols = ['city', 'date', 'match_type', 'player_of_match', 'toss_winner', 'toss_decision', 'umpire1', 'umpire2']
    for col in metadata_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Master Dictionary: Unify historical team names and abbreviations
    team_mappings = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'RCB': 'Royal Challengers Bengaluru',
        'CSK': 'Chennai Super Kings',
        'MI': 'Mumbai Indians',
        'KKR': 'Kolkata Knight Riders',
        'SRH': 'Sunrisers Hyderabad',
        'RR': 'Rajasthan Royals',
        'DC': 'Delhi Capitals',
        'PBKS': 'Punjab Kings',
        'LSG': 'Lucknow Super Giants',
        'GT': 'Gujarat Titans'
    }
    df['batting_team'] = df['batting_team'].replace(team_mappings)
    df['bowling_team'] = df['bowling_team'].replace(team_mappings)

# Master Dictionary: Unify ALL fractured venue names (including newer/secondary stadiums)
    venue_mappings = {
        # Bengaluru
        'M Chinnaswamy Stadium': 'M.Chinnaswamy Stadium, Bengaluru',
        'M Chinnaswamy Stadium, Bengaluru': 'M.Chinnaswamy Stadium, Bengaluru',
        'M.Chinnaswamy Stadium': 'M.Chinnaswamy Stadium, Bengaluru',
        # Chennai
        'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chennai',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium, Chennai',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium, Chennai',
        # Hyderabad
        'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Hyderabad',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium, Hyderabad',
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium, Hyderabad',
        # Delhi
        'Feroz Shah Kotla': 'Arun Jaitley Stadium, Delhi',
        'Arun Jaitley Stadium': 'Arun Jaitley Stadium, Delhi',
        # Mumbai
        'Wankhede Stadium': 'Wankhede Stadium, Mumbai',
        'Brabourne Stadium': 'Brabourne Stadium, Mumbai',
        'Dr DY Patil Sports Academy': 'Dr DY Patil Sports Academy, Mumbai',
        # Mohali / Punjab
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
        'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur, Chandigarh',
        'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur, New Chandigarh': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur, Chandigarh',
        # Ahmedabad
        'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium, Ahmedabad',
        
        # --- THE NEW & SECONDARY STADIUM FIXES ---
        # Dharamsala
        'Himachal Pradesh Cricket Association Stadium': 'Himachal Pradesh Cricket Association Stadium, Dharamsala',
        # Visakhapatnam (Vizag)
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
        # Pune
        'Maharashtra Cricket Association Stadium': 'Maharashtra Cricket Association Stadium, Pune',
        'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium, Pune',
        # Raipur
        'Shaheed Veer Narayan Singh International Stadium': 'Shaheed Veer Narayan Singh International Stadium, Raipur',
        # Jaipur
        'Sawai Mansingh Stadium': 'Sawai Mansingh Stadium, Jaipur',
        # Lucknow
        'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow'
    }
    
    df['venue'] = df['venue'].replace(venue_mappings)

    # ==========================================
    # FEATURE ENGINEERING & TRAINING
    # ==========================================
    print("Engineering venue features...")
    innings_scores = df.groupby(['match_id', 'inning', 'venue'])['total_runs'].sum().reset_index()
    venue_stats = innings_scores.groupby('venue').agg(
        venue_avg_score=('total_runs', 'mean'),
        match_count=('match_id', 'nunique')
    ).reset_index()
    venue_stats = venue_stats[venue_stats['match_count'] >= 5]
    df = pd.merge(df, venue_stats[['venue', 'venue_avg_score']], on='venue', how='inner')

    print("Engineering live match context features...")
    df = df.sort_values(by=['match_id', 'inning', 'over', 'ball']).reset_index(drop=True)
    df['current_score'] = df.groupby(['match_id', 'inning'])['total_runs'].cumsum()
    df['is_wicket'] = df['player_dismissed'].notna().astype(int)
    df['wickets_fallen'] = df.groupby(['match_id', 'inning'])['is_wicket'].cumsum()
    
    df['balls_bowled'] = (df['over'] - 1) * 6 + df['ball']
    df['balls_left'] = 120 - df['balls_bowled']
    df['wickets_left'] = 10 - df['wickets_fallen']
    df['crr'] = (df['current_score'] * 6) / np.maximum(df['balls_bowled'], 1)

    df['is_powerplay'] = (df['over'] <= 6).astype(int)
    df['is_death'] = (df['over'] >= 16).astype(int)
    
    final_scores = df.groupby(['match_id', 'inning'])['current_score'].last().reset_index()
    final_scores.columns = ['match_id', 'inning', 'final_total_score']
    df = pd.merge(df, final_scores, on=['match_id', 'inning'], how='inner')

    print("Preparing datasets for training...")
    features = [
        'batting_team', 'bowling_team', 'venue', 'current_score', 
        'balls_left', 'wickets_left', 'venue_avg_score', 
        'crr', 'is_powerplay', 'is_death'
    ]
    X = df[features]
    y = df['final_total_score']
    
    X = pd.get_dummies(X, columns=['batting_team', 'bowling_team', 'venue'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Optimized XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nTraining Complete! Mean Absolute Error (MAE): {mae:.2f} runs")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_predictor.pkl')
    joblib.dump(X.columns.tolist(), 'models/model_features.pkl')
    print("Model artifacts successfully saved to models/ folder.")

if __name__ == "__main__":
    train_math_brain()