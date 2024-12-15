from flask import Flask, request, jsonify, render_template
from process_file import process_uploaded_file
from loadTheModelOnce import load_everything
import os
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# LOADING THE MODEL IN THE BACKGROUND
model, tokenizer, feature_extractor = None, None, None
def load_model():
    global model, tokenizer, feature_extractor
    print("Loading the model...")
    model, tokenizer, feature_extractor = load_everything()
    print("Model loaded!")
background_thread = threading.Thread(target=load_model, daemon=True)
background_thread.start()

# ROUTE ONE
@app.route('/')
@app.route('/index')
@app.route('/home')
def home():
    return render_template('index.html');

# ROUTE TWO
@app.route('/captioning', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        print("There is a POST Request")
        if 'file1' not in request.files:
            return jsonify({'message': 'There is no file1 in the form!'}), 400
        
        file1 = request.files['file1']
        if file1.filename == '':
            return jsonify({'message': 'No file selected!'}), 400

        # Save the uploaded file
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        print('The Image has been saved locally!')

        # Call the external function and get its result
        try:
            print("Check if the model is loaded")
            while(model == None):
                time.sleep(5)
            print("The model is successfully loaded in the memory! Moving Forward")
            print('Retrieving the caption from model')
            result = process_uploaded_file(path, model, tokenizer, feature_extractor)
            return jsonify({'message': result})
        except Exception as e:
            return jsonify({'message': f'Error during processing: {e}'}), 500
    
    return render_template("captioning.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)