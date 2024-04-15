#import pickle
#from sklearn.model_selection import train_test_split
#import pandas as pd
#import numpy as np
#from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import sys
#from sklearn.metrics import ConfusionMatrixDisplay
from diagnostics import model_predictions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

log_file = os.path.join(os.getcwd(), config['log_file'])
dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])

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

y_pred, y_test = model_predictions()
##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"), dpi=300, transparent=False, bbox_inches='tight')

    logging.info(f"[ DATA: Confusion matrix saved to '{output_model_path}/confusionmatrix.png' ]")


if __name__ == '__main__':
    score_model()
