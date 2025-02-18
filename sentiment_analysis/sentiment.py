from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io
import base64
import eventlet
from transformers import pipeline
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


nltk.download('punkt')

sentiment_model = pipeline(model="AshBunny/finetuning-sentiment-model-5000-samples")

eventlet.monkey_patch()

app = Flask(__name__)
CORS(app)


# socketio = SocketIO(app, cors_allowed_origins="*")

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

def generate_sentiment(text):
    res = sentiment_model([text])
    ans = ""
    if res[0]["label"] == "LABEL_1":
        ans = "positive"
    else:
        ans = "negative"
    
    return ans


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
    

@app.route('/text_update', methods=['POST'])
def handle_text_update():
    """Generate a new word cloud and return it as a JSON response."""
    data = request.get_json()  # Ensure the request contains JSON data
    text = data.get("text", "")

    wordcloud_image = generate_wordcloud(text)
    sentiment = generate_sentiment(text)
    bar_image = generate_barchart(text)

    return jsonify({
        "image": wordcloud_image,
        "image1": bar_image,
        "sentiment": sentiment
    })


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5002)
