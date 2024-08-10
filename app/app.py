from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# URL of the FastAPI inference API
INFERENCE_API_URL = "http://fastapi:8000/predict"

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handle requests to the root URL ('/') of the Flask application.
    
    On a GET request, the function renders an HTML form where users can input text for processing.
    On a POST request, the function:
    - Retrieves the input text from the submitted form.
    - Sends the input text to the FastAPI inference API for POS and NER prediction.
    - Receives the predicted tags from the FastAPI API.
    - Renders the HTML template with the input text, input tokens, predicted POS tags, and NER tags.
    
    Returns:
        A rendered HTML page with input text and prediction results (if available).
    """
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['input_text']

        # Send the input text to the FastAPI inference API
        response = requests.post(INFERENCE_API_URL, json={"text": input_text})

        # Get the predicted tags from the API response
        if response.status_code == 200:
            result = response.json()
            input_tokens = result['input_tokens']
            pos_tags = ','.join(result['predicted_pos_tags'])
            ner_tags = ','.join(result['predicted_ner_tags'])

            return render_template('index.html', input_text=input_text, input_tokens=input_tokens, pos_tags=pos_tags, ner_tags=ner_tags)

    # On GET request, just render the form
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
