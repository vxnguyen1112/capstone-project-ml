from flask import Flask, request,jsonify
import pandas as pd
import numpy as np
from joblib import load


app = Flask(__name__)

@app.route("/")
def home_view():
        return "<h1>Hello World!</h1>"
@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    df_test = pd.DataFrame(columns=list(data.keys()))
    
    df_test.loc[0] = np.array(list(data.values()))

    # Load pre-trained model
    clf = load(str("./saved_model/random_forest.joblib"))
    
    
    predicted=clf.predict(df_test)
    result = {'result':predicted.item(0)}
    
    return jsonify(result)