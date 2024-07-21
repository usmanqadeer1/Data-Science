import streamlit as st
import re
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
import joblib  # For loading tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image


# Load the saved tokenizer
tokenizer = joblib.load('tokenizer1.joblib')  # Replace with your tokenizer file path

# Load the saved model
model = tf.keras.models.load_model('sentiment_model1.h5')  # Replace with your model file path


# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt')
max_sequence_length = 100

# Function to preprocess the text data
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)     # Remove mentions
    text = re.sub(r'#\S+', '', text)     # Remove hashtags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters and punctuation
    text = text.lower()                  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])  # Remove stop words
    return text


def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')  # Ensure max_sequence_length matches your model
    prediction = model.predict(padded_sequence)
    return prediction


# Main function to run the app
def main():
    # Page title and icon
    st.set_page_config(page_title='Tweet Sentiment Analyzer', page_icon=":speech_balloon:")

    # Display title and description
    st.title('Tweet Sentiment Analyzer')
    st.write('Enter a tweet to analyze its sentiment.')

    # Input text box for entering tweet
    tweet = st.text_input('Input Tweet:')

    # Button to analyze sentiment
    if st.button('Analyze'):
        if tweet.strip() == '':
            st.warning('Please enter a tweet.')
        else:
            sentiment = predict_sentiment(tweet).argmax()
            # Color-coded output based on sentiment
            if sentiment == 0:
                st.success(f'Tweet sentiment: Irrelevant')
            elif sentiment == 1:
                st.info(f'Tweet sentiment: Negative')
            elif sentiment == 2:
                st.error(f'Tweet sentiment: Neutral')
            elif sentiment == 3:
                st.error(f'Tweet sentiment: Positive')

# Run the main function
if __name__ == '__main__':
    main()
