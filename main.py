import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import joblib
import pickle

# Load the trained model
model = load_model('LC_CNN_Model.keras')

# Load the tokenizer
# tokenizer = joblib.load('LIM_CNN_Tokenizer.pkl')

with open('LC_Tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('LC_label_encoder.pkl', 'rb') as l:
    label_encoder = pickle.load(l)
    

# Define the maximum sequence length
max_length = 100  

# Define a function to preprocess input text
def preprocess_input(text):
    # Tokenize input text
    sequences = tokenizer.texts_to_sequences([text])
    processed_text = pad_sequences(sequences, maxlen=max_length)
    return processed_text

# Define a function to predict language
def predict_language(text):
    processed_text = preprocess_input(text)
    predictions = model.predict(processed_text)
    return predictions


class_names = ['Arabic', 'Chinese', 'Dutch', 'English', 'Estonian', 'French',
       'Greek', 'Hindi', 'Indonesian', 'Japanese', 'Kannada', 'Korean',
       'Latin', 'Malayalam', 'Persian', 'Portugese', 'Pushto', 'Romanian',
       'Russian', 'Spanish', 'Swedish', 'Tamil', 'Thai', 'Turkish',
       'Urdu']
# label_encoder.classes_

# Sidebar
st.sidebar.title("Language Identification Model")
st.sidebar.image("language_icon.jpg", use_column_width=True)
st.sidebar.markdown("""
    ### How It Works
    1. **Enter Text:** Type or paste a text in any language.
    2. **Prediction:** Our system will predict the language of the input text.
    3. **Results:** View the predicted language.
""")

# Main Page
st.title("Language Identification Model")

# Input text
input_text = st.text_area("Enter Text:", "")

# Prediction button
if st.button("Predict"):
    if input_text:
        predicted_language = predict_language(input_text)
        predicted_index = np.argmax(predicted_language)
        # Get the class name corresponding to the predicted index
        predicted_class = class_names[predicted_index]
        st.success("Predicted Language: {}".format(predicted_class))
    else:
        st.warning("Please enter some text for prediction.")
