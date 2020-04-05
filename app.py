from flask import Flask, request, jsonify, render_template
from flask_news_classifier.build_model import *

app = Flask(__name__)

news = NewsDector()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['POST'])
def predict():
    try:
        text = request.form['message']
        number = news.predict(text)
        print(number)
        return render_template('result.html', prediction=number)
    except Exception as e:
        print(e)


if __name__ == '__main__':
	app.run(debug=True,host="0.0.0.0",port=80)