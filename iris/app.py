from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        SepalLengthCm = float(request.form['SepalLengthCm'])
        SepalWidthCm = float(request.form['SepalWidthCm'])
        PetalLengthCm = float(request.form['PetalLengthCm'])
        PetalWidthCm = float(request.form['PetalWidthCm'])


        values = np.array([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
        prediction = model.predict(values)


        return render_template('index.html', prediction_text='Hasilnya adalah {}'.format(prediction))





if __name__ == "__main__":
    app.run(debug=True)

