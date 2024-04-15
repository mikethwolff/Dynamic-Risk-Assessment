"""
Note: 

If you run this file:

    $ python app.py

Then, you can curl the results on another terminal:

    $ curl http://localhost:8000/scoring
    $ curl http://localhost:8000/summarystats
    $ curl http://localhost:8000/diagnostics/timing

Otherwise run: 
    
    $ python apicalls.py
"""

from flask import Flask, request
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])

with open(os.path.join(output_model_path, 'trainedmodel.pkl'), "rb") as model:
    prediction_model = pickle.load(model)

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filepath = request.json.get('filepath')
    y_test, pred = model_predictions(file_path=filepath)
    
    #add return value for prediction outputs
    #print("\n" + str(pred) + "\n")
    return str(f"{pred}")

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():      
    #check the score of the deployed model
    score = score_model()
    
    #add return value (a single F1 score number)
    #print("\n" + str(score) + "\n")
    return str(f"[ METRICS: F1={score} ]")

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    statistics = dataframe_summary()

    #return a list of all calculated summary statistics
    #print("\n" + str(statistics) + "\n")
    return str(f"[ STATISTICS: \n{statistics} ]") 

#######################Diagnostics Endpoint
@app.route("/diagnostics/timing", methods=['GET','OPTIONS'])
def timing():  
    timing = execution_time()
    response = str(f"[ TIMING: \nExecution time for ingestion.py: {timing[0]} \nExecution time for training.py: {timing[1]} ]")

    #add return value for all diagnostics
    #print("\n" + response + "\n")
    return response

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
