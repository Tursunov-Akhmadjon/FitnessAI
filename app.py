import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Health Predictor",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for Ghibli style
def add_ghibli_style():
    st.markdown("""
    <style>
    /* Main Theme Colors */
    :root {
        --ghibli-green: #7AA874;
        --ghibli-blue: #A0C3D2;
        --ghibli-beige: #F7F1E5;
        --ghibli-pink: #F8CBA6;
    }
    
    /* Background and Base Styling */
    .stApp {
        background-color: var(--ghibli-beige);
        background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%237aa874' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E");
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #3D5656;
        font-family: 'Arial Rounded MT Bold', sans-serif;
    }
    
    h1 {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .prediction-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin: 10px 0;
        border: 2px solid var(--ghibli-green);
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    
    /* Input Fields */
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 12px;
    }
    
    /* Button Styling */
    .stButton button {
        background-color: var(--ghibli-green);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #689f63;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Section Divider */
    hr {
        border: 0;
        height: 3px;
        background-image: linear-gradient(to right, rgba(122, 168, 116, 0), rgba(122, 168, 116, 0.75), rgba(122, 168, 116, 0));
        margin: 30px 0;
    }
    
    /* Results Section */
    .results-container {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--ghibli-green);
    }
    
    .metric-label {
        font-size: 1.2rem;
        color: #3D5656;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #666;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)


# # Function to create Ghibli-style clouds image
def create_clouds_image(width=800, height=200):
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    
    # Drawing would be done here with PIL
    # For demo purposes, we return a blank transparent image
    
    # Convert to base64 for displaying in the app
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f'data:image/png;base64,{img_str}'

# Gender Image Selection
def get_avatar(gender):
    if gender == "Male":
        return "assets/male.png"
    elif gender == "Female":
        return "assets/female.png"
    return "assets/neutral.png"

# Main function
def main():
    add_ghibli_style()
    
    # Title with Ghibli-style clouds
    st.markdown(f"""
    <div style='text-align: center;'>
        <img src="{create_clouds_image()}" style="width: 100%; height: 100px; object-fit: contain; margin-bottom: -50px;">
        <h1>✨ Health Journey Predictor ✨</h1>
        <p style='font-size: 1.2rem; margin-bottom: 30px;'>Discover your optimal sleep hours and daily calorie needs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    sleep_model = joblib.load("app/Sleep_hours.joblib")
    calorie_model = joblib.load("app/calories_model.joblib")
    

    # Create columns for form layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h2>✏️ Your Health Information</h2>", unsafe_allow_html=True)
        
        # Create form
        with st.form("health_form"):
            # Personal details
            st.markdown("<h3>Personal Details</h3>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            
            with col_a:
                age = st.number_input("Age", value=None, placeholder="Type a number...")
                gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select...",)
                
            with col_b:
                height = st.number_input("Height (cm)", value=None, placeholder="Type a number...")
                weight = st.number_input("Weight (kg)", value=None, placeholder="Type a number...")
                
            
            # Workout details
            st.markdown("<h3>Workout Information</h3>", unsafe_allow_html=True)
            col_c, col_d = st.columns(2)
            

            with col_c:
                workout_type = st.selectbox("Workout Type", ["Cardio", "Strength", "Yoga", "HIIT", "Running"])
                duration = st.number_input("Workout Duration (mins)", value=None, placeholder="Type a number...")
                calories_burned = st.number_input("Calories Burned", value=None, placeholder="Type a number...")
                heart_rate = st.number_input("Heart Rate (bpm)", value=None, placeholder="Type a number...")
            
            with col_d:
                steps = st.number_input("Steps Taken", value=None, placeholder="Type a number...")
                distance = st.number_input("Distance (km)", value=None, placeholder="Type a number...")
                intensity = st.selectbox("Workout Intensity", ["Low", "Medium", "High"])
                resting_hr = st.number_input("Resting Heart Rate (bpm)", value=None, placeholder="Type a number...")
            
            # Intensity and mood
            st.markdown("<h3>Intensity & Mood</h3>", unsafe_allow_html=True)
            col_e, col_f = st.columns(2)
            
            with col_e:
                mood_before = st.select_slider("Mood Before Workout", options=["Tired", "Happy", "Neutral", "Stressed"])

            
            with col_f:
                mood_after = st.select_slider("Mood After Workout", options=["Fatigued", "Neutral", "Energized"])


            # Submit button
            submit_button = st.form_submit_button("Predict My Health Metrics")
    
    # Show Ghibli-inspired image in the right column
    with col2:
        # image_path = f"assets/{'male' if gender == 'male' else 'female'}.png"
        # avatar = Image.open(image_path)
        avatar = Image.open(get_avatar(gender))
        st.image(avatar, use_container_width=True)
    
        # Process form and display predictions
        if submit_button:
                        
            input_features = [
                age, gender, height, weight, workout_type, duration, calories_burned, heart_rate,
                steps, distance, intensity, resting_hr, mood_before, mood_after
            ]

            columns = ['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'Workout Type', 'Workout Duration (mins)', 
                    'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken', 'Distance (km)', 'Workout Intensity', 
                    'Resting Heart Rate (bpm)', 'Mood Before Workout', 'Mood After Workout']
            
            # Sleep Prediction
            # Create DataFrame
            input1_df = pd.DataFrame([input_features], columns=columns)
            predicted_sleep = sleep_model.predict(input1_df)[0]
            if predicted_sleep < 4.0:
                predicted_sleep = 4.0

            # Add sleep_hours to input for calorie model
            insert_index = columns.index('Workout Intensity') + 1
            # Insert into both lists
            columns.insert(insert_index, 'Sleep Hours')
            input_features.insert(insert_index, predicted_sleep)

            # Create DataFrame
            input2_df = pd.DataFrame([input_features], columns=columns)
            
            predicted_calories = calorie_model.predict(input2_df)[0]
            
            # Display results
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h2>🔮 Your Personalized Health Predictions</h2>", unsafe_allow_html=True)
            
            col_results1, col_results2 = st.columns(2)
            
            with col_results1:
                st.markdown("""
                <div class="prediction-card">
                    <h3>Recommended Sleep Hours</h3>
                    <div class="metric-value">%.1f hours</div>
                    <p>Based on your age, workout intensity, and other metrics, this is your optimal sleep duration for recovery.</p>
                </div>
                """ % predicted_sleep, unsafe_allow_html=True)
                
            with col_results2:
                st.markdown("""
                <div class="prediction-card">
                    <h3>Daily Calorie Intake</h3>
                    <div class="metric-value">%d calories</div>
                    <p>This is your recommended daily calorie intake based on your metrics and optimal sleep hours.</p>
                </div>
                """ % predicted_calories, unsafe_allow_html=True)
        
        # # Additional information
        # st.markdown("<hr>", unsafe_allow_html=True)
        # st.markdown("""
        # <div style="padding: 20px; background-color: rgba(255, 255, 255, 0.7); border-radius: 15px;">
        #     <h3>✨ Health Insights</h3>
        #     <p>These predictions are personalized based on your unique profile. Keep in mind that your needs may change with your activity level and other factors.</p>
            
        #     <p>For optimal health:</p>
        #     <ul>
        #         <li>Stay hydrated throughout the day</li>
        #         <li>Maintain a balanced diet with plenty of vegetables</li>
        #         <li>Listen to your body and adjust your routine as needed</li>
        #     </ul>
        # </div>
        # """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>✨ Health Journey Predictor - Created with Streamlit and Ghibli-inspired design ✨</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
