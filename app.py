from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import re
# Ensure nltk punkt tokenizer is available

app = Flask(__name__)

# Load Hugging Face model & tokenizer
# Load model directly
tokenizer = AutoTokenizer.from_pretrained("TheLongTran/ChatBot-WithRLHF")
model = AutoModelForCausalLM.from_pretrained("TheLongTran/ChatBot-WithRLHF")

conversation_history = []

def split_sentences(text):
    """Splits text into sentences using regex instead of nltk."""
    return re.split(r"(?<=[.!?])\s+", text.strip())

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])


def chat():
    global cnt  # Make cnt persistent
    cnt = 0
    global conversation_history 
    conversation_history = [] # Use global history
    user_input = request.form["user_input"]
    
    # Keep conversation history (short-term memory)
    conversation_history.append(user_input)
    if len(conversation_history) > 5:  
        conversation_history.pop(0)

    input_list = " ".join(conversation_history)
    input_text = f"{input_list}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    first_sentence = split_sentences(response)
    conversation_history.append(first_sentence)
    print(cnt)  # Store bot's response in memory
    if cnt < 3:
        cnt += 1
    print(response)
    return jsonify({"response": first_sentence[cnt]})

if __name__ == "__main__":
    app.run(debug=True)
