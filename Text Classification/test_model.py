import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.models import load_model 
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)

# Load EMBER V1 model
def ember_v1_model():
    model = SentenceTransformer('llmrails/ember-v1')
    return model

# Load pre-trained response data
import json
with open("responses.json", "r") as f:
    response_data = json.load(f)

# Load the pre-trained ANN model

from keras.models import model_from_json
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
ann_model = model_from_json(loaded_model_json)
ann_model.load_weights("customer_model.weights.h5")
print("Loaded Model from disk")
ann_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print("Successfully loaded the ANN model")

# Load EMBER V1 model
model = ember_v1_model()
print("Successfully loaded the EMBER V1 model")

@app.route('/batch_predict', methods=['GET'])
def batch_predict():
    test_cases = [
        "I want to cancel the order", 
        "Let me know how to change my address", 
        "List me all the payment methods that you will follow to process a payment", 
        "working hours for customer support",
        "delivery status", 
        "how to get a refund", 
        "how to write a review", 
        "how to change to another account"
    ]
    
    test_final_array = np.array([model.encode(each_one) for each_one in test_cases])
    
    pred_labels = ann_model.predict(test_final_array)
    predicted_indices = np.argmax(pred_labels, axis=1)
    
    predictions = {}
    for idx in predicted_indices:
        predictions[idx] = response_data.get(str(idx), "No response found")
    
    return jsonify(predictions)

@app.route('/predict_intent', methods=['POST'])
def custom_input_test():
    data = request.get_json()
    input_case = data.get("text", "")
    
    test_final_array = np.array([model.encode(input_case)])
    
    pred_labels = ann_model.predict(test_final_array)
    predicted_indices = np.argmax(pred_labels, axis=1)
    
    response = response_data.get(str(predicted_indices[0]), "No response found")
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5004)
