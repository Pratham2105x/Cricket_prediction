import streamlit as st
import pickle
import pandas as pd
import joblib
model = joblib.load("pipe.joblib")

st.title("🏏 IPL Match Win Probability Predictor")

# Valid IPL teams
teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

# Full venue list (cleaned from your dataset)
venues = [
    'Arun Jaitley Stadium', 'Barabati Stadium', 'Barsapara Cricket Stadium, Guwahati',
    'Brabourne Stadium', 'Buffalo Park', 'De Beers Diamond Oval',
    'Dr DY Patil Sports Academy', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
    'Dubai International Cricket Stadium', 'Eden Gardens',
    'Eden Gardens, Kolkata', 'Feroz Shah Kotla', 'Himachal Pradesh Cricket Association Stadium',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala', 'Holkar Cricket Stadium',
    'JSCA International Stadium Complex', 'Kingsmead', 'MA Chidambaram Stadium',
    'MA Chidambaram Stadium, Chepauk', 'MA Chidambaram Stadium, Chepauk, Chennai',
    'M Chinnaswamy Stadium', 'M Chinnaswamy Stadium, Bengaluru',
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Maharashtra Cricket Association Stadium',
    'Maharashtra Cricket Association Stadium, Pune',
    'Narendra Modi Stadium, Ahmedabad', 'New Wanderers Stadium', 'Newlands',
    'OUTsurance Oval', 'Punjab Cricket Association IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
    'Punjab Cricket Association Stadium, Mohali', 'Rajiv Gandhi International Stadium',
    'Rajiv Gandhi International Stadium, Uppal',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
    'Sardar Patel Stadium, Motera', 'Sawai Mansingh Stadium',
    'Sawai Mansingh Stadium, Jaipur', 'Shaheed Veer Narayan Singh International Stadium',
    'Sharjah Cricket Stadium', 'Sheikh Zayed Stadium', "St George's Park",
    'Subrata Roy Sahara Stadium', 'SuperSport Park', 'Vidarbha Cricket Association Stadium, Jamtha',
    'Wankhede Stadium', 'Wankhede Stadium, Mumbai',
    'Zayed Cricket Stadium, Abu Dhabi'
]

# Team and venue input
batting_team = st.selectbox("Batting Team", sorted(teams))
bowling_team = st.selectbox("Bowling Team", sorted([t for t in teams if t != batting_team]))
venue = st.selectbox("Match Venue", sorted(venues))

# Match stats input
target = st.number_input("Target Score")
current_score = st.number_input("Current Score")
overs = st.number_input("Overs Completed", step=0.1)
wickets = st.slider("Wickets Lost", 0, 10)

# Feature engineering
runs_left = target - current_score
balls_left = 120 - (overs * 6)
wicket_left = 10 - wickets
curr_run_rate = current_score / overs if overs > 0 else 0
req_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0

if st.button("Predict Win Probability"):
    input_df = pd.DataFrame([{
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'venue': venue,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wicket_left': wicket_left,
        'runs_total_y': target,
        'curr_run_rate': curr_run_rate,
        'req_run_rate': req_run_rate
    }])

    try:
        result = model.predict_proba(input_df)[0]
        win = result[1] * 100
        lose = result[0] * 100

        st.subheader("📊 Predicted Win Probability")
        st.success(f"🏆 Batting Team Win Chance: {win:.2f}%")
        st.error(f"🛡️ Bowling Team Win Chance: {lose:.2f}%")
    except Exception as e:
        st.warning("⚠️ Prediction failed. Please ensure all values are valid and try again.")
        st.text(f"Error: {e}")
