import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app layout
st.title('Linear Regression Model Deployment')

# Input features from the user
# You should define inputs for all 14 features here
feature1 = st.number_input('Feature 1:')
feature2 = st.number_input('Feature 2:')
feature3 = st.number_input('Feature 3:')
feature4 = st.number_input('Feature 4:')
feature5 = st.number_input('Feature 5:')
feature6 = st.number_input('Feature 6:')
feature7 = st.number_input('Feature 7:')
feature8 = st.number_input('Feature 8:')
feature9 = st.number_input('Feature 9:')
feature10 = st.number_input('Feature 10:')
feature11 = st.number_input('Feature 11:')
feature12 = st.number_input('Feature 12:')
feature13 = st.number_input('Feature 13:')
feature14 = st.number_input('Feature 14:')

# Prediction button
if st.button('Predict'):
    # Prepare the input data as a 2D array with all 14 features
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5,
                             feature6, feature7, feature8, feature9, feature10,
                             feature11, feature12, feature13, feature14]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the result
    st.success(f'The predicted value is: {prediction[0]}')
