import os
import json
import random
from flask import Flask, render_template, request, jsonify
import torch
import requests

app = Flask(__name__)

# Load intents.json
with open('intents.json') as json_data:
    intents = json.load(json_data)


API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer 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"}

def load_intents():
    """Load intents from a JSON file."""
    with open('intents.json') as json_data:
        return json.load(json_data)
    
def query_huggingface(payload):
    """Send a query to the Hugging Face API and return the response."""
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTPError: {e.response.text}")
        return None

def get_response(user_text, intents):
    """Get a response based on user text and predefined intents."""
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_text.lower():
                return random.choice(intent["responses"])
    return "I do not understand..."

@app.route("/")
def index_get():
    """Render the base HTML template."""
    return render_template("base.html")

@app.route("/predict", methods=['POST'])
def predict():
    text = request.get_json().get("message")
    intents = load_intents()  # Reload or reuse the global intents variable
    response = get_response(text, intents)
    if response == "I do not understand...":
        # Construct the payload for Hugging Face API
        payload = {"inputs": text}
        api_response = query_huggingface(payload)
        if api_response and isinstance(api_response, list) and api_response[0]:
            # Assuming the response structure is correct, extract the generated text
            response = api_response[0].get("generated_text", "Sorry, I couldn't process your request.")
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
