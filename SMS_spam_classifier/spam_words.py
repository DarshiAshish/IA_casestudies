from flask import Flask, render_template, request, jsonify
from wordcloud import WordCloud
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io
import base64
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
import pickle
import re

app = Flask(__name__)
CORS(app)
# Spam word list
spam_words = [
    "win", "free", "prize", "cash", "gift", "claim", "offer", "congratulations",
    "discount", "urgent", "limited", "act now", "special promotion", "exclusive", 
    "million", "guaranteed", "investment", "earn", "hurry", "call now", "winner", "spam"
]

def find_spam_words(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    detected_spam_words = [word for word in words if word in spam_words]
    return detected_spam_words

def generate_wordcloud(text):
    wordcloud = WordCloud(width=500, height=300, background_color='white').generate(text)
    
    img_buffer = io.BytesIO()
    plt.figure(figsize=(5, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close()

    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def generate_barchart(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in punctuations]
    word_counts = Counter(filtered_tokens)
    x = list(word_counts.keys())
    y = list(word_counts.values())
    img_buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.bar(x, y, color='lightblue')  # Using light blue color for the bars
    plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotate labels for readability
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Bar Chart (Excluding Stop Words and Punctuation)")
    plt.tight_layout()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close()
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# Load the spam classifier model and feature extractor
model = pickle.load(open('svc.sav', 'rb'))
feature_extraction = pickle.load(open('feature_extraction.pkl', 'rb'))

def spam(text):
    input_data_features = feature_extraction.transform([text])
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        return "Ham"
    else:
        return "Spam"


@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Summarize the provided text and return word cloud, bar chart, and spam detection results."""
    data = request.get_json()
    text = data.get("text", "")

    wordcloud_image = generate_wordcloud(text)
    bar_image = generate_barchart(text)

    detected_spam_words = find_spam_words(text)
    spam_result = spam(text)

    return jsonify({
         "spam_result": spam_result,
        "possible_spam_words": detected_spam_words,
        "wordcloud_image": wordcloud_image,
        "bar_image": bar_image
       
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5010)
