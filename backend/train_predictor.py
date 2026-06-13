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
    
    # Standardize & Filter Noise
    print("Filtering noise...")
    if 'method' in df.columns:
        df = df[df['method'] != 'D/L']
        
    # Unify team names across historical seasons
    team_mappings = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Rising Pune Supergiant': 'Rising Pune Supergiants'
    }
    df['batting_team'] = df['batting_team'].replace(team_mappings)
    df['bowling_team'] = df['bowling_team'].replace(team_mappings)

    # Feature Engineering — Venue Historical Baselines
    print("Engineering venue features...")
    innings_scores = df.groupby(['match_id', 'inning', 'venue'])['total_runs'].sum().reset_index()
    
    # Get the average score per venue
    venue_stats = innings_scores.groupby('venue').agg(
        venue_avg_score=('total_runs', 'mean'),
        match_count=('match_id', 'nunique')
    ).reset_index()
    
    # Filter for stadiums that hosted at least 5 matches to keep statistical relevance
    venue_stats = venue_stats[venue_stats['match_count'] >= 5]
    
    # Merge these macro baselines back into our main dataframe
    df = pd.merge(df, venue_stats[['venue', 'venue_avg_score']], on='venue', how='inner')

    # Feature Engineering — Live Match Context
    print("Engineering live match context features...")
    # Sort chronologically
    df = df.sort_values(by=['match_id', 'inning', 'over', 'ball']).reset_index(drop=True)
    
    # 1. Current Running Score
    df['current_score'] = df.groupby(['match_id', 'inning'])['total_runs'].cumsum()
    
    # 2. Wickets Fallen
    df['is_wicket'] = df['player_dismissed'].notna().astype(int)
    df['wickets_fallen'] = df.groupby(['match_id', 'inning'])['is_wicket'].cumsum()
    
    # 3. Time Assets Remaining (Out of 120 balls in T20)
    df['balls_bowled'] = (df['over'] - 1) * 6 + df['ball']
    df['balls_left'] = 120 - df['balls_bowled']
    df['wickets_left'] = 10 - df['wickets_fallen']
    
    # NEW FEATURE: 4. Current Run Rate (CRR)
    # Using np.maximum to prevent division by zero on the very first ball
    df['crr'] = (df['current_score'] * 6) / np.maximum(df['balls_bowled'], 1)

    # NEW FEATURE: 5. Game Phase Flags
    df['is_powerplay'] = (df['over'] <= 6).astype(int)
    df['is_death'] = (df['over'] >= 16).astype(int)
    
    # 6. Define the Target Variable (What the model must predict)
    final_scores = df.groupby(['match_id', 'inning'])['current_score'].last().reset_index()
    final_scores.columns = ['match_id', 'inning', 'final_total_score']
    df = pd.merge(df, final_scores, on=['match_id', 'inning'], how='inner')

    # Data Preparation & Categorical Encoding
    print("Preparing datasets for training...")
    
    # UPDATED FEATURES LIST
    features = [
        'batting_team', 'bowling_team', 'venue', 'current_score', 
        'balls_left', 'wickets_left', 'venue_avg_score', 
        'crr', 'is_powerplay', 'is_death'
    ]
    X = df[features]
    y = df['final_total_score']
    
    # Convert text strings to columns of 1s and 0s
    X = pd.get_dummies(X, columns=['batting_team', 'bowling_team', 'venue'])
    
    # Split: 80% to train the model, 20% held back to test performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training, Evaluation, & Serialization
    print("Training Optimized XGBoost Regressor...")
    
    # TUNED HYPERPARAMETERS
    model = xgb.XGBRegressor(
        n_estimators=400,        # Build 400 trees instead of 100
        learning_rate=0.05,      # Learn half as fast
        max_depth=7,             # Allow the trees to look slightly deeper
        subsample=0.8,           # Train each tree on 80% of the data to prevent overfitting
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nTraining Complete! Mean Absolute Error (MAE): {mae:.2f} runs")
    
    # Save the output binaries
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_predictor.pkl')
    joblib.dump(X.columns.tolist(), 'models/model_features.pkl')
    print("Model artifacts successfully saved to models/ folder.")

if __name__ == "__main__":
    train_math_brain()