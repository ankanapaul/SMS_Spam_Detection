from flask import Flask, render_template, request
import joblib
from spam_detector import preprocess_text  

app = Flask(__name__)

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = preprocess_text(text)  
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)