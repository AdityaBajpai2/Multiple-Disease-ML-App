import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# Loading the saved models in binary mode:

loaded_model = pickle.load(open("C:/Users/ADITYA/Downloads/ML_Project/trained_file.sav",'rb'))

pinkersons_model = pickle.load(open("C:/Users/ADITYA/Downloads/ML_Project/pirnkersons_file.sav",'rb'))

heart_model = pickle.load(open("C:/Users/ADITYA/Downloads/ML_Project/heart_file.sav",'rb'))

cancer_model = pickle.load(open("C:/Users/ADITYA/Downloads/ML_Project/cancer_prediction.sav",'rb')) 

# Prediction function for diabetes:
def diabetes_pred(input_data):
    # convert text inputs to floats (empty -> 0.0) and reshape to 2D for model
    nums = [float(x) if x != "" else 0.0 for x in input_data]
    arr = np.array(nums).reshape(1, -1)
    pred = loaded_model.predict(arr)
    if pred[0] == 1:
        return 'The person is Diabetic.'
    else:
        return 'The person is not Diabetic.'


#Prediction function for Breast cancer:
def cancer_pred(input_array_reshaped):
    nums = [float(x) if x != "" else 0.0 for x in input_array_reshaped]
    arr = np.array(nums).reshape(1, -1)
    pred = cancer_model.predict(arr)
    if pred[0] == 1:
        return "The person has Cancer."
    else:
        return "The person is Cancer Free."

#Prediction for Pinkerson's Disease:
def pinkerson_pred(input_array):
    nums = [float(x) if x != "" else 0.0 for x in input_array]
    arr = np.array(nums).reshape(1, -1)
    pred = pinkersons_model.predict(arr)
    if pred[0] == 1:
        return 'The person is positive with Pinkersons.'
    else:
        return 'The person is Healthy.'

#Prediction function for heart disease:
def heart_pred(input_array):
    nums = [float(x) if x != "" else 0.0 for x in input_array]
    arr = np.array(nums).reshape(1, -1)
    pred = heart_model.predict(arr)
    if pred[0] == 1:
        return 'The person has Heart Disease.'
    else:
        return 'The person is Healthy.'


#Creating the side-bar for navigation:
    
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System', 
                           ['Diabetes Prediction',
                            'Pinkersons Prediction',
                            'Heart Disease Prediction',
                            'Breast Cancer Prediction'],
                           icons = ['clipboard2-pulse','bag-heart','activity'],
                           default_index = 0)

# Diabetes Prediction Page:
    
if selected == "Diabetes Prediction":
    
    #Page title
    st.title("Diabetes Prediction using ML")
    
    #Getting the input from the user:
    
    #columns for input fields:
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Enter the number of pregnancies if any ->")
    
    with col2:
        Glucose = st.text_input("Enter your blood glucose level here -> ")
    
    with col3:
        BloodPressure = st.text_input("Enter your current blood pressure here ->")
    
    with col1:
        SkinThickness = st.text_input("Enter the thickenss of your skin here -> ")
    
    with col2:
        Insulin = st.text_input("Enter your insulin level here -> ")
    
    with col3:
        BMI = st.text_input("Enter your BMI here -> ")
    
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value -> ")
    
    with col2:
        Age = st.text_input("Enter your age here -> ")

    #code for making a prediction:
    diab_diagnosis = ''
    
    #Creating a button for prediction:
    if st.button("Diabetes Result"):
        diab_diagnosis = diabetes_pred(
            [Pregnancies, Glucose, BloodPressure, SkinThickness,
             Insulin, BMI, DiabetesPedigreeFunction, Age]
        )
    st.success(diab_diagnosis)
    
    
#Pinkerson'S Prediction Page:

elif selected == "Pinkersons Prediction":
    
    #Page title
    st.title("Pinkersons Prediction using ML")
    
    #Getting the input from the user:
    
    #columns for input fields:
    col1,col2,col3 = st.columns(3)
    
    with col1:
        mdvp_fo_hz = st.text_input("Enter MDVP:Fo(Hz) here -> ")
    
    with col2:
        mdvp_fhi_hz = st.text_input("Enter MDVP:Fhi(Hz) here ->")
    
    with col3:
        mdvp_flo_hz = st.text_input("Enter MDVP:FLO(Hz) here ->")
        
    with col1:
        mdvp_jitter_percent = st.text_input("Enter MMDVP:Jitter(%) here ->")
       
    with col2:
        mdvp_jitter_abs = st.text_input("Enter your MDVP:Jitter(Abs) here ->")

    with col3:
        mdvp_rap = st.text_input("Enter MDVP:RAP here ->")
    
    with col1:
        mdvp_ppq = st.text_input("Enter MDVP:PPQ here ->")
        
    with col2:
        mdvp_shimmer = st.text_input("Enter MDVP:Shimmer here ->")
    
    with col3:
        mdvp_shimmer_db = st.text_input("Enter MDVP:Shimmer(dB) here ->")

    with col1:
        shimmer_apq_three = st.text_input("Enter Shimmer:APQ3 here ->")
         
    with col2:
        shimmer_aqp_five = st.text_input("Enter Shimmer:APQ5 here ->")
    
    with col3:
        mdvp_apq = st.text_input("Enter MDVP:APQ here ->")
    
    with col1:
        shimmer_dda = st.text_input("Enter Shimmer:DDA here ->")
    
    with col2:
        nhr = st.text_input("Enter NHR here ->")
    
    with col3:
        hnr = st.text_input("Enter HNR here ->")

    with col1:
        status = st.text_input("Enter Status here ->")
         
    with col2:
        rpde = st.text_input("Enter RPDE here ->")
    
    with col3:
        dfa = st.text_input("Enter DFA here ->")
    
    with col1:
        spread_one = st.text_input("Enter spread1 ->")
    
    with col2:
        spread_two = st.text_input("Enter spread2 here ->")
    
    with col3:
        d_two = st.text_input("Enter D2 here ->")
    
    with col1:
        ppec = st.text_input("Enter PPEc here ->")
        
    #code for making a prediction:
    pinkersons_diagnosis = ''
    
    #Creating a button for prediction:
    if st.button("Pinkerson's Prediction Result"):
        pinkersons_diagnosis = pinkerson_pred([
            mdvp_fo_hz, mdvp_fhi_hz, mdvp_flo_hz,
            mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap,
            mdvp_ppq, mdvp_shimmer, mdvp_shimmer_db,
            shimmer_apq_three, shimmer_aqp_five, mdvp_apq,
            shimmer_dda, nhr, hnr, status, rpde,
            dfa, spread_one, spread_two, d_two, ppec
        ])

    st.success(pinkersons_diagnosis)
    
#Breast Cancer Page:

elif selected == "Breast Cancer Prediction":  
    
    #Page title
    st.title("Breast Cancer Prediction using ML")
    
    #Getting the inputs from the user:
    col1, col2, col3 = st.columns(3)
        
    with col1:
        radius_mean = st.text_input("Enter radius mean here ->")
    
    with col2:
        texture_mean = st.text_input("Enter texture mean here ->")
    
    with col3:
        perimeter_mean = st.text_input("Enter perimeter mean here ->")
        
    with col1:
        area_mean = st.text_input("Enter area mean here ->")
    
    with col2:
        smoothness_mean = st.text_input("Enter smootheness mean here ->")
    
    with col3:
        compactness_mean = st.text_input("Enter compactness mean here ->")
        
    with col1:
        concavity_mean = st.text_input("Enter concavity mean here ->")
    
    with col2:
        concave_points_mean = st.text_input("Enter concave points mean here ->")
    
    with col3:
        symmetry_mean = st.text_input("Enter symmetry mean here ->")
        
    with col1:
        fractal_dimension_mean = st.text_input("Enter fractal dimension mean here ->")
    
    with col2:
        radius_se = st.text_input("Enter radius_se here ->")
    
    with col3:
        texture_se = st.text_input("Enter texture_se here ->")
            
    with col1:
        perimeter_se = st.text_input("Enter perimeter_se here ->")
    
    with col2:
        area_se = st.text_input("Enter area_se here ->")
    
    with col3:
        smoothness_se = st.text_input("Enter smoothness_se here ->")
         
    with col1:
        compactness_se = st.text_input("Enter compactness_se here ->")
    
    with col2:
        concavity_se = st.text_input("Enter concavity_se here ->")
    
    with col3:
        symmetry_se = st.text_input("Enter symmetry_se here ->")
        
    with col1:
        fractal_dimension_se = st.text_input("Enter fractal_dimension_se here ->")
    
    with col2:
        radius_worst = st.text_input("Enter radius_worst here ->")
    
    with col3:
        texture_worst = st.text_input("Enter texture_worst here ->")

    with col1:
        perimeter_worst = st.text_input("Enter perimeter_worst here ->")
    
    with col2:
        area_worst = st.text_input("Enter area_worst here ->")
    
    with col3:
        smoothness_worst = st.text_input("Enter smoothness_worst here ->")
        
    with col1:
        compactness_worst = st.text_input("Enter compactness_worst here ->")
    
    with col2:
        concavity_worst = st.text_input("Enter concavity_worst here ->")
    
    with col3:
        concave_points_worst = st.text_input("Enter concave_pointsworst here ->")
            
    with col1:
        symmetry_worst = st.text_input("Enter symmetry_worst here ->")
    
    with col2:
        fractal_dimension_worst = st.text_input("Enter fractal_dimension_worst here ->")
    
    # code for making a prediction:
    cancer_diagnosis = ""

    #Creating a button for prediction:
    if st.button("Cancer Prediction Result"):
        cancer_diagnosis = cancer_pred([
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
            fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
            smoothness_se, compactness_se, concavity_se, symmetry_se,
            fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
            area_worst, smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst
        ])

    st.success(cancer_diagnosis)

#Heart Disease Page:

elif selected == "Heart Disease Prediction":

    #Page title
    st.title("Heart Disease Prediction using ML")
    
    #Getting the input from the user:
    
    #columns for input fields:
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input("Enter your age here -> ")
    
    with col2:
        sex = st.text_input("Enter your gender here ->")
    
    with col3:
        cp = st.text_input("Enter CP here ->")
        
    with col1:
        trestbps = st.text_input("Enter trestbps here ->")
       
    with col2:
        chol = st.text_input("Enter your cholestrol level here ->")

    with col3:
        fbs = st.text_input("Enter fbs here ->")
    
    with col1:
        restecg = st.text_input("Enter restecg here ->")
        
    with col2:
        thalach = st.text_input("Enter thalach here ->")
    
    with col3:
        exang = st.text_input("Enter exang here ->")

    with col1:
        oldpeak = st.text_input("Enter oldpeak here ->")
         
    with col2:
        slope = st.text_input("Enter slope here ->")
    
    with col3:
        ca = st.text_input("Enter ca here ->")
    
    with col1:
        thal = st.text_input("Enter thal here ->")
        
    #code for making a prediction:
    heart_diagnosis = ''
    
    #Creating a button for prediction:
    if st.button("Heart Prediction Result"):
        heart_diagnosis = heart_pred(
            [age, sex, cp, trestbps, chol, fbs,
             restecg, thalach, exang, oldpeak, slope, ca, thal]
        )

    st.success(heart_diagnosis)