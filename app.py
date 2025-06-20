import streamlit as st
import pickle
import numpy as np

# Load your model
with open('match_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏏 Cricket Match Win Probability Predictor (2nd Innings)")

target = st.number_input("Target Score")
current = st.number_input("Current Score")
overs = st.number_input("Overs Completed", step=0.1)
wickets = st.slider("Wickets Lost", 0, 10)

crr = current / overs if overs > 0 else 0
rrr = (target - current) / (20 - overs) if overs < 20 else 0

if st.button("Predict Chances"):
    X = np.array([[target, current, overs, wickets, crr, rrr]])
    proba = model.predict_proba(X)[0]
    
    win = proba[1] * 100
    lose = proba[0] * 100

    st.subheader("📊 Win Probability")
    st.success(f"✅ Win Chance: {win:.2f}%")
    st.error(f"❌ Lose Chance: {lose:.2f}%")
