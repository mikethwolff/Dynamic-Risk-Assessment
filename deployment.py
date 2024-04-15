import os
import json
from shutil import copy2
import logging
import sys

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

log_file = os.path.join(os.getcwd(), config['log_file'])
# dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
artifacts_path = os.path.join(config['output_model_path'])
ingestedfiles_path = os.path.join(config["output_folder_path"])


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

####################function for deployment
def store_model_into_pickle(model):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    """ Copy the artifacts to prod deployment directory """
    for filename in os.listdir(model):
        copy2(
            os.path.join(
                artifacts_path, filename), os.path.join(
                prod_deployment_path, filename))

    copy2(
        os.path.join(
            ingestedfiles_path,
            'ingestedfiles.txt'),
        os.path.join(
            prod_deployment_path,
            'ingestedfiles.txt'))

    logging.info("[ FILE: Artifacts copied to deployment directory ]")


if __name__ == '__main__':
    store_model_into_pickle(artifacts_path)
