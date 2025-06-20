import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# 🎯 Load Trained Model (Random Forest with joblib)
model = joblib.load("pipe.joblib")

# -----------------------------
# 🏏 App Title
st.title("🏏 IPL Match Win Probability Predictor")

# -----------------------------
# 📋 Define IPL Teams and Venues (must match training data)
teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

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

# -----------------------------
# 🧮 Team and Venue Selections
batting_team = st.selectbox("Batting Team", sorted(teams))
bowling_team = st.selectbox("Bowling Team", sorted([t for t in teams if t != batting_team]))
venue = st.selectbox("Match Venue", sorted(venues))

# -----------------------------
# 🔢 Match Stat Inputs (Validated)
target = st.number_input("🎯 Target Score", min_value=1, step=1, format="%d")
current_score = st.number_input("🔢 Current Score", min_value=0, step=1, format="%d")
over_input = st.text_input("⏱ Overs Completed (e.g., 10.3, 14.5)", value="0.0")
wickets = st.slider("🚑 Wickets Lost", 0, 10)

# -----------------------------
# 🔁 Convert Overs to Total Balls (10.3 = 10 overs + 3 balls = 63 balls)
try:
    over_parts = over_input.split(".")
    overs_int = int(over_parts[0])
    balls_in_over = int(over_parts[1]) if len(over_parts) > 1 else 0
    if balls_in_over > 5:
        st.error("⚠️ Invalid over format. Use .0 to .5 only.")
        st.stop()
    total_balls = overs_int * 6 + balls_in_over
    overs = total_balls / 6
except:
    st.error("⚠️ Overs must be in correct decimal format like 14.2")
    st.stop()

# -----------------------------
# 🧠 Feature Engineering
runs_left = target - current_score
balls_left = 120 - total_balls
wicket_left = 10 - wickets
curr_run_rate = current_score / overs if overs > 0 else 0
req_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0

# -----------------------------
# 🔮 Prediction Button
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
        # 🧨 All-Out Condition Handling (No need to call model)
        if wickets == 10:
            st.subheader("📊 Predicted Win Probability")
            st.success("🏆 Batting Team Win Chance: 0.00%")
            st.error("🛡️ Bowling Team Win Chance: 100.00%")
        else:
            # 🔍 Model Prediction
            result = model.predict_proba(input_df)[0]
            win = result[1] * 100
            lose = result[0] * 100

            # 🎯 Display Result
            st.subheader("📊 Predicted Win Probability")
            st.success(f"🏆 Batting Team Win Chance: {win:.2f}%")
            st.error(f"🛡️ Bowling Team Win Chance: {lose:.2f}%")
    except Exception as e:
        st.warning("⚠️ Prediction failed. Please check input values.")
        st.text(f"Error: {e}")
