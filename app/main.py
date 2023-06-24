from flask import Flask, request,jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from joblib import load
import logging


app = Flask(__name__)
CORS(app)

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
@app.route('/stroke', methods=['POST'])
def stroke():
    model = load('./saved_model/stroke-prediction-model.joblib')    
    arr=request.get_json()
    worktype= arr['worktype'] #Work Type

    arr8=[0] #Govt Job
    arr9=[0] #Never_worked
    arr10=[0] #Private
    arr11=[0] #Self-Employed
    arr12=[0] #Children
    
    if worktype == 0:
       arr8=[1] #Govt Job
       arr9=[0] #Never_worked
       arr10=[0] #Private
       arr11=[0] #Self-Employed
       arr12=[0] #Children
    elif worktype == 1:
        arr8=[0] #Govt Job
        arr9=[1] #Never_worked
        arr10=[0] #Private
        arr11=[0] #Self-Employed
        arr12=[0] #Children
    elif worktype == 2:
        arr8=[0] #Govt Job
        arr9=[0] #Never_worked
        arr10=[1] #Private
        arr11=[0] #Self-Employed
        arr12=[0] #Children
    elif worktype == 3:
        arr8=[0] #Govt Job
        arr9=[0] #Never_worked
        arr10=[0] #Private
        arr11=[1] #Self-Employed
        arr12=[0] #Children
    elif worktype == 4:
        arr8=[0] #Govt Job
        arr9=[0] #Never_worked
        arr10=[0] #Private
        arr11=[1] #Self-Employed
        arr12=[0] #Children

    residencetype= arr['residencetype'] #Residence Type

    arr13=[0] #Rural
    arr14=[0] #Urban
    
    if residencetype == 0:
        arr13=[1] #Rural
        arr14=[0] #Urban
    elif residencetype == 1:
        arr13=[0] #Rural
        arr14=[1] #Urban
        
    smokingtype= arr['smokingtype'] #Smoking Type

    arr15=[0] #Formerly Smoked
    arr16=[0] #Never Smoked
    arr17=[0] #Smokes
    
    
    if smokingtype == 0:
        arr15=[1] #Formerly Smoked
        arr16=[0] #Never Smoked
        arr17=[0] #Smokes
    elif smokingtype == 1:
        arr15=[0] #Formerly Smoked
        arr16=[1] #Never Smoked
        arr17=[0] #Smokes
    elif smokingtype == 2:
        arr15=[0] #Formerly Smoked
        arr16=[0] #Never Smoked
        arr17=[1] #Smokes
    all_values = [value for value in arr.values()]
    array = np.concatenate((all_values[0:7], arr8, arr9, arr10, arr11, arr12, arr13, arr14, arr15, arr16, arr17), axis=0)
    app.logger.info(array)  # Log message using app.logger
    predictionOutcome = model.predict([array])
    result = {'result':predictionOutcome.item(0)}
    return jsonify(result)