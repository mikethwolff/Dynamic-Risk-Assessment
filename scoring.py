import pandas as pd
import pickle
import os
from sklearn import metrics
import json
import logging
import sys

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

score_log = os.path.join(os.getcwd(), config['score_log'])
dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

output_model_path = os.path.join(config['output_model_path'])


# Define Logging
# Define logging handlers
logFileHandler = logging.FileHandler(score_log)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logFileHandler,
        consoleHandler
    ]
)

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    with open(os.path.join(output_model_path, 'trainedmodel.pkl'), "rb") as file:
        model = pickle.load(file)

    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = df.loc[:, df.columns != 'corporation']
    y_test = X_test.pop('exited').values.reshape(-1, 1).ravel()

    y_pred =  model.predict(X_test)

    f1_score = metrics.f1_score(y_pred, y_test)

    # Logging is writing the result to latestscore.txt (defined in config.json)
    logging.info(f"[ METRICS: F1={f1_score:.2f} ]")

    return f1_score

if __name__ == '__main__':
    score_model()
