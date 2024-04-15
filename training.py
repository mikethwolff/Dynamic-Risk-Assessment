import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
import json
import logging
import sys


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

log_file = os.path.join(os.getcwd(), config['log_file'])
dataset_csv_path = os.path.join(config['output_folder_path'])
#model_path = os.path.join(config['output_model_path'])
output_model_path = config['output_model_path']
model_filename = config['model_filename']
model_path = os.path.join(os.getcwd(), config['output_model_path'], config['model_filename'])
random_seed = config['random_seed']
test_size = config['test_size']
#features = config['features']


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

#################Function for training the model
def train_model():

    #use this logistic regression for training
    model = LogisticRegression(
        C=1.0, class_weight=None, 
        dual=False, 
        fit_intercept=True,
        intercept_scaling=1, 
        l1_ratio=None, 
        max_iter=100,
        #multi_class='warn', 
        n_jobs=None, 
        penalty='l2',
        random_state=0, 
        solver='liblinear', 
        tol=0.0001, 
        verbose=0,
        warm_start=False)

    #fit the logistic regression to your data
    logging.info("[ DATA: Loading data... ]")
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    # x contains the predictor variables, in our case all, except the first column
    # Alternative: X = df[features] or X = df.iloc[:, 1:] or X = df.drop('corporation', axis=1)
    X_train = df.loc[:, df.columns != 'corporation']

    # y is the target/promoted variable
    # Alternative: y = df["exited"]
    y_train = X_train.pop('exited').values.reshape(-1, 1).ravel()

    # Train
    logging.info("[ MODEL: Training model... ]")
    model.fit(X_train.values, y_train)
    #model.fit(X_train, y_train)
    logging.info("[ FINISHED: Training successful ]")

    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(model_path,'wb'))
    logging.info(f"[ FILE: Model saved to '{output_model_path}/{model_filename}' ]")


if __name__ == '__main__':
    train_model()
