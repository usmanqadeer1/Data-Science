import streamlit as st
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib  # For loading tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image


# Load the saved tokenizer
tokenizer = joblib.load('tokenizer1.joblib')  # Replace with your tokenizer file path
# tokenizer = Tokenizer(num_words=20000) 

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
    words = word_tokenize(text)
    text = ' '.join([word for word in words if word not in stop_words])  # Remove stop words
    return text


def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)  # Ensure max_sequence_length matches your model
    prediction = model.predict(padded_sequence)
    return prediction[0][0]


# Main function to run the app
def main():
    # Page title and icon
    tweet = "I am so happy today"
    sentiment = predict_sentiment(tweet)
    print(sentiment)
    # Color-coded output based on sentiment
    if sentiment == 0:
        print(f'Tweet sentiment: Irrelevant')
    elif sentiment == 1:
        print(f'Tweet sentiment: Negative')
    elif sentiment == 2:
        print(f'Tweet sentiment: Neutral')
    elif sentiment == 3:
        print(f'Tweet sentiment: Positive')

# Run the main function
if __name__ == '__main__':
    main()
