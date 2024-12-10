import json
import numpy as np
import pandas as pd
import os
import torch
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from keras import models
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
import re
from transformers import AutoTokenizer, BertTokenizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Initialize Keras tokenizer


#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = keras.preprocessing.text.Tokenizer(num_words=31924)  # Adjust number of words as needed



# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path and disaster names path relative to the current file
model_path = os.path.join(BASE_DIR, 'res', 'modelLSTM.h5')
disaster_names_path = os.path.join(BASE_DIR, 'res', 'disasterNames.json')

csv_path =  os.path.join(BASE_DIR, 'res', 'train.csv')
df = pd.read_csv(csv_path)
texts = df['text'].values.tolist()
labels = df['target'].values.tolist()
tokenizer.fit_on_texts(texts)

# Load disaster names
with open(disaster_names_path, 'r') as file:
    disaster_names = json.load(file)

model = models.load_model(model_path)
print("Model loaded successfully!")


@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()
        input_data = data.get('input_text', '')
        print(f"input text: {input_data}")
        
        if not input_data:
            return jsonify({"error": "No input text provided"}), 400
        
        # Preprocess and pad the input
        padded_data = preprocess_input(input_data)

        # Get prediction from the model
        prediction = model.predict(padded_data)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction values: {prediction}")

        # Assuming binary classification, convert the prediction to a class (0 or 1)
        predicted_class = 1 if prediction[-1][0] > 0.5 else 0
        print(f"Predicted class: {predicted_class}")


        # Process the input text to match disaster keywords
        input_words = set(input_data.lower().split())
        matched_disasters = input_words.intersection(disaster_names)
        
        if matched_disasters:
            return jsonify({'prediction': predicted_class, 'matched_disasters': list(matched_disasters)})
        else:
            return jsonify({'prediction': predicted_class})
    
    except Exception as e:
        print(e)
        return jsonify({"errorMessage": f"An error occurred: {str(e)}"}), 500

def tokenize_and_prepare(data, max_length=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data = data.split(' ')
    tokens = [tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True) for text in data]
    token_sequences = keras.preprocessing.sequence.pad_sequences(tokens, maxlen=max_length, padding='post')
    return token_sequences

def preprocess_input(input_text):
    # Preprocess the input text as you did earlier
    # Example: Remove unwanted characters, stopwords, lemmatize, etc.
    preprocessed_text = preprocess_text(input_text)
    print(preprocessed_text)
    # Convert the preprocessed text into sequences using the fitted tokenizer
    #encoded_data = tokenizer.texts_to_sequences([preprocessed_text])
    #padded_data = keras.preprocessing.sequence.pad_sequences(encoded_data, padding='post', maxlen=128)
    padded_data = tokenize_and_prepare(preprocessed_text)
    return padded_data

def preprocess_text(text):
    # Remove links, special characters, mentions, hashtags, and numbers
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    stopword = set(stopwords.words('english'))
    extrastopwords = set(["im", "like", "get", "dont", "wont", "via", "still", "would", "got", "rt", "cant", "theyre", "bb", "fyi", "hmu", "th", "st", "rd"])
    text = " ".join([word for word in text.split() if word not in stopword and word not in extrastopwords])

    # Lemmatization with POS tagging
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    text_lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(text_lemmas)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

if __name__ == '__main__':
    app.run(debug=True)
