import pandas as pd
import timeit
import os
import json
import logging
import sys
import pickle
import subprocess
from sklearn.metrics import fbeta_score, precision_score, recall_score


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

log_file = os.path.join(os.getcwd(), config['log_file'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = os.path.join(config['output_folder_path'])
finaldata = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))

# Define Logging
# Define logging handlers
logFileHandler = logging.FileHandler(log_file)
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

##################Function to get model predictions
def model_predictions(dir_path = test_data_path, file_path = 'testdata.csv'):
    #read the deployed model and a test dataset, calculate predictions

    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), 'rb') as model:
        model = pickle.load(model)

    # Read data
    testdata = pd.read_csv(os.path.join(dir_path, file_path))
    X_test = testdata.iloc[:, 1:]
    y_test = X_test.pop('exited').values.reshape(-1, 1).ravel()

    pred = model.predict(X_test.values)

    precision, recall, fbeta = compute_model_metrics(y_test, pred)
    logging.info(f"[ METRICS: Precision: {precision}, Recall: {recall}, Fbeta: {fbeta} ]")

    #return value should be a list containing all predictions
    return y_test, pred

def compute_model_metrics(y, preds):
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here

    numeric = finaldata.select_dtypes(include='int64')
    statistics = numeric.iloc[:, :-1].agg(['mean', 'median', 'std'])

    logging.info(f"[ STATISTICS: \n{statistics} ]")

    nas = list(finaldata.isna().sum())
    na_percentage = (f"{[nas[i] / len(finaldata.index) for i in range(len(nas))]}%")

    logging.info(f"[ STATISTICS: Percentage of NA in columns: {na_percentage} ]")

    #return value should be a list containing all summary statistics
    return statistics 

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py

    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    timing_ingestion = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.system('python training.py')
    timing_training = timeit.default_timer() - start_time

    # Execution time saved to logs/churn.log
    logging.info(f"[ TIMING: Execution time for ingestion.py: {timing_ingestion} ]")
    logging.info(f"[ TIMING: Execution time for training.py: {timing_training} ]")

    #return a list of 2 timing values in seconds
    return [timing_ingestion, timing_training] 

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    # Outdated dependencies saved to logs/churn.log
    logging.info(f"[ FILE: Outdated dependencies: \n{outdated} ]")

    return str(outdated)

if __name__ == '__main__':
    execution_time()
    model_predictions()
    dataframe_summary()
    outdated_packages_list()
