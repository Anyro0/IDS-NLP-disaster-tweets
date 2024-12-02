from flask import Flask, request, jsonify
from transformers import pipeline  # Assuming you are using Hugging Face's Transformers for text classification

# Initialize Flask app and load the trained NLP model
app = Flask(__name__)

nlp_model = pipeline("text-classification", model="your-trained-model-path")  # Replace with actual model

@app.route('/')
def index():
    return "NLP Disaster Tweets API"

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = nlp_model(text)
    
    prediction_label = result[0]['label']
    
    return jsonify({"prediction": prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
