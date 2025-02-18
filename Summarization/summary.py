from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Loading the Extractive model 
from summarizer import Summarizer
extractive_model = Summarizer()

# Loading the abstractive model
from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')



ex_length = 0.75

def generate_extractive_text(text):
    res = extractive_model(text, ratio = ex_length)
    res = ''.join(res)
    return res, len(res.split(" "))

def generate_abstractive_text(text):
    res = summarizer(text, max_length=len(text.split(" ")), min_length=int(len(text.split(" "))/2), do_sample=False)
    return res[0]['summary_text'], len(res[0]['summary_text'].split(" "))

def similarity_metric(text1, text2):
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity


@app.route('/summary', methods=["POST"])
def home():
    data = request.get_json()  
    text = data.get("text", "")
    ext_res, ext_length = generate_extractive_text(text)
    gen_res, gen_length = generate_abstractive_text(text)
    similarity = similarity_metric(ext_res, gen_res)
    return jsonify({
        "extractive_summary": ext_res,
        "extractive_length": ext_length,
        "abstractive_summary": gen_res,
        "abstractive_length": gen_length,
        "similarity": float(similarity),
        "original_length": len(text.split(" "))
    })

    
if __name__ == '__main__':
    app.run(debug=True, port=5011)
    
