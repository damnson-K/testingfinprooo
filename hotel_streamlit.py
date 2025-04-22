import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.graph_objects as go
from PIL import Image
from models import AdjustedThresholdModel

# Set page config
st.set_page_config(page_title="Job Intention Predictor", page_icon="🏃‍♀️💨", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    with open('final_model.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Function to convert date to week number
def get_week_number(date):
    return date.isocalendar()[1]

# Function to create gauge chart
def create_gauge_chart(probability):
    # Detect the current theme
    is_dark_theme = st.get_option("theme.base") == "dark"
    
    # Set colors based on the theme
    bg_color = "rgba(0,0,0,0)" if is_dark_theme else "white"
    text_color = "white" if is_dark_theme else "black"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Leave after Training Probability", 'font': {'size': 24, 'color': text_color}},
        number = {'font': {'size': 20, 'color': text_color}},
        gauge = {
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': text_color},
            'bar': {'color': "#4CAF50"},
            'bgcolor': bg_color,
            'borderwidth': 2,
            'bordercolor': text_color,
            'steps': [
                {'range': [0, 0.41], 'color': "#90EE90"},
                {'range': [0.41, 1], 'color': "#FF6347"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7}}))
    
    fig.update_layout(
        paper_bgcolor = bg_color,
        font = {'color': text_color, 'family': "Arial"},
        height = 300,
        margin = dict(l=10, r=10, t=50, b=10)
    )
    return fig

# Header
header_image = Image.open("contoso_header.png")
st.image(header_image, use_column_width=True)

# Navigation
col1, col2, col3 = st.columns([1,2,1])
with col1:
    st.write("")  # This is to create some space
# with col2:
    # st.title("Job Intention Predictor")
with col3:
    predictor_button = st.button("Predictor")
    about_button = st.button("About Us")

# Main content
if about_button:
    st.title("About Us")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .team-member {
        font-size:18px !important;
        margin-left: 20px;
    }
    .about-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-container">', unsafe_allow_html=True)
    st.markdown('<p class="big-font">This Job Intention Predictor was created by Beta Consulting:</p>', unsafe_allow_html=True)
    st.markdown('<p class="team-member">🧑‍💻 <a href="https://www.linkedin.com/in/kerin-m/" target="_blank" style="text-decoration: none; color: inherit;">Kerin Mulianto</a></p>', unsafe_allow_html=True)
    st.markdown('<p class="team-member">👨‍💻 <a href="https://www.linkedin.com/in/timothy-hartanto/" target="_blank" style="text-decoration: none; color: inherit;">Timothy Hartanto</a></p>', unsafe_allow_html=True)
    st.markdown('<p class="team-member">🧑‍💻 <a href="https://www.linkedin.com/in/wafanabilas/" target="_blank" style="text-decoration: none; color: inherit;">Wafa Nabila</a></p>', unsafe_allow_html=True)


else:  # Default to Predictor page
    st.title("Job Intention Predictor")

    # Input form
    st.subheader("📝 Enter Enrollee Details")

    col1, col2 = st.columns(2)

    # Categorical Features
    with col1:
        city = st.selectbox("City", options=['city_103', 'city_40', 'city_21', 'city_115', 'city_162',
       'city_176', 'city_160', 'city_46', 'city_61', 'city_114',
       'city_13', 'city_159', 'city_102', 'city_67', 'city_100',
       'city_16', 'city_71', 'city_104', 'city_64', 'city_101', 'city_83',
       'city_105', 'city_73', 'city_75', 'city_41', 'city_11', 'city_93',
       'city_90', 'city_36', 'city_20', 'city_57', 'city_152', 'city_19',
       'city_65', 'city_74', 'city_173', 'city_136', 'city_98', 'city_97',
       'city_50', 'city_138', 'city_82', 'city_157', 'city_89',
       'city_150', 'city_70', 'city_175', 'city_94', 'city_28', 'city_59',
       'city_165', 'city_145', 'city_142', 'city_26', 'city_12',
       'city_37', 'city_43', 'city_116', 'city_23', 'city_99', 'city_149',
       'city_10', 'city_45', 'city_80', 'city_128', 'city_158',
       'city_123', 'city_7', 'city_72', 'city_106', 'city_143', 'city_78',
       'city_109', 'city_24', 'city_134', 'city_48', 'city_144',
       'city_91', 'city_146', 'city_133', 'city_126', 'city_118',
       'city_9', 'city_167', 'city_27', 'city_84', 'city_54', 'city_39',
       'city_79', 'city_76', 'city_77', 'city_81', 'city_131', 'city_44',
       'city_117', 'city_155', 'city_33', 'city_141', 'city_127',
       'city_62', 'city_53', 'city_25', 'city_2', 'city_69', 'city_120',
       'city_111', 'city_30', 'city_1', 'city_140', 'city_179', 'city_55',
       'city_14', 'city_42', 'city_107', 'city_18', 'city_139',
       'city_180', 'city_166', 'city_121', 'city_129', 'city_8',
       'city_31', 'city_171'])
        relevent_experience = st.selectbox("Relevant Experience", options=['Has relevent experience', 'No relevent experience'])
        enrolled_university = st.selectbox("Type of Current University Enrollment", options=['no_enrollment', 'Full time course', 'Part time course'])
        education_level = st.selectbox("Highest Education Level", options=['Graduate', 'Masters', 'High School', 'Phd', 'Primary School'])
        major_discipline = st.selectbox("Major Discipline", options=['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major', 'Other'])
        experience_bin = st.selectbox("Total Years of Experience", options=['<1', '1-5', '6-10', '11-15', '16-20', '>20'])
        company_size = st.selectbox("Previous Company Size", options=['missing', '<10','10-49','50-99','100-500','500-999','1000-4999','5000-9999','>10000'])
        company_type = st.selectbox("Previous Company Type", options=['Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Other', 'Public Sector', 'NGO'])
        last_new_job_bin = st.selectbox("Years Spent in Previous Job", options=['<=1', '2-3', '>=4'])

    # Numerical Features
    with col2:
        city_development_index = st.slider("City Development Index (CDI):", min_value=0.44, max_value=0.95, value=100.0, step=0.01)


    # Prepare input data
    input_data = pd.DataFrame({
        'city': [city],
        'relevent_experience': [relevent_experience],
        'enrolled_university': [enrolled_university],
        'education_level': [education_level],
        'major_discipline': [major_discipline],
        'experience_bin': [experience_bin],
        'company_size': [company_size],
        'company_type': [company_type],
        'last_new_job_bin': [last_new_job_bin],
        'city_development_index': [float(city_development_index)],
    })

    # Make prediction
    if st.button("'Leave after Training' Probability Prediction"):
        try:
            prediction = model.predict_proba(input_data)[0][1]
            
            # Create and display gauge chart
            fig = create_gauge_chart(prediction)
            st.plotly_chart(fig, use_container_width=True)
            
            # Determine risk level and background color
            if prediction > 0.41:
                leave_risk = "Leave"
                bg_color = "rgba(144, 238, 144, 0.3)"
            else:
                leave_risk = "Stay"
                bg_color = "rgba(255, 99, 71, 0.3)"  
            
            # Create a container with the background color
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                        <h3 style="text-align: center;">{leave_risk} risk of leaving after training: {prediction:.2%}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Input data:")
            st.write(input_data)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Created by Beta Consulting using Streamlit</p>", unsafe_allow_html=True)