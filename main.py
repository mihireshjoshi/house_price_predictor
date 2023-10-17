from flask import Flask, render_template, request, json
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

dt_model = pickle.load(open('house_price_predictor_dt.pkl','rb'))

app = Flask(__name__,static_url_path='/static')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['GET', 'POST'])
def ml():
    if request.method == 'POST':
        try:
            # Get input values from the HTML form
            sqft = float(request.form.get("sqft"))
            lo = float(request.form.get("lo"))
            la = float(request.form.get("la"))
            bhk = int(request.form.get("bhk"))
            rera = float(request.form.get("rera"))
            brk = int(request.form.get("brk"))
            rom = int(request.form.get("rom"))
            resale = int(request.form.get("resale"))
            uc = int(request.form.get("uc"))

            # Prepare the input data as a NumPy array
            input_data = np.array([uc,rera,bhk,brk,sqft,rom,resale,lo,la]).reshape(1, -1)

            # Call your ML model to make predictions
            prediction = dt_model.predict(input_data)

            # You can return the prediction as JSON
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('results.html', result = prediction + " Lakhs")

if __name__ == '__main__':
    app.run(debug=True)
