"""
Note: 

Start local server on another console:

    $ python app.py

"""
from training import train_model
from diagnostics import model_predictions
import logging
import sys
import os
import json
import ingestion
from sklearn.metrics import f1_score

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

log_file = os.path.join(os.getcwd(), config['log_file'])
input_path = os.path.join(config['input_folder_path'])
prod_directory = os.path.join(config['prod_deployment_path'])
output_folder_path = config['output_folder_path']
artifacts_path = os.path.join(config['output_model_path'])
control_file_path = os.path.join(output_folder_path, "ingestedfiles.txt")
input_folder_path = config['input_folder_path']


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

##################Check and read new data
#first, read ingestedfiles.txt
log_entry = []
current_files = []
with open(os.path.join(prod_directory, 'ingestedfiles.txt')) as file:
    log_entry = file.read().splitlines()

# Extracting file names using string manipulation
logged_files = [entry.split(',')[1].split(':')[1].strip().strip() for entry in log_entry]

print("\n[ Logged files:" + str(logged_files) + " ]\n")

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
with open(control_file_path, "a+") as control_file:
    # For any given path look for subdirectories and collect filenames
    for root, _, files in os.walk(input_folder_path):
        for eachFileName in files:
            if eachFileName.endswith(".csv"):
                current_files.append(eachFileName)

print("\n[ Current files:" + str(current_files) + " ]\n")

added_files = [file for file in current_files if file not in logged_files]
#print("Added files:", added_files if added_files else "No files added.")

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if added_files:
    ingestion.merge_multiple_dataframe()
    print(f"[ Added files: {added_files} ]")
else:
    print("[ No files added ]")
    exit(0)

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_directory, 'latestscore.txt')) as file:
    old_f1_score = str(file.read())

pred, y_test = model_predictions(output_folder_path, 'finaldata.csv')
new_f1_score = str(f1_score(pred, y_test))

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
logging.info("[ MODEL DRIFT: Comparing scores... ]")
if new_f1_score >= old_f1_score:
    logging.info("[ RESULT: No model drift detected ]")
    exit(0)

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
logging.info("[ RESULT: Model drift detected ]")
logging.info("[ MODEL: Retraining model... ]")
train_model()
logging.info("[ FINISHED: Model re-trained successfully ]")

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system("python3 reporting.py")
os.system("python3 apicalls.py")
