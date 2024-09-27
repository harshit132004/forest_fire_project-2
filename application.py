import pickle
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import ridge regressor and standard scaler pickle files
ridge_model = pickle.load(open('models/ridgecv.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    # Loads the landing page (index.html)
    return render_template("index.html")

@app.route("/predict")
def predict():
    # Loads the prediction form (home.html)
    return render_template("home.html")

@app.route("/predictdata", methods=['POST'])
def predict_datapoint():
    if request.method == "POST":
        # Get data from the form in home.html
        data = [float(request.form[key]) for key in ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region','BUI','DC']]
        
        # Transform input data using the scaler
        new_data = standard_scaler.transform([data])
        
        # Make prediction using the ridge model
        result = ridge_model.predict(new_data)
        
        # Pass the prediction result to home.html
        return render_template('home.html', result=result[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)






   
