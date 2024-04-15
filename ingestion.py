import pandas as pd
import os
import json
from datetime import datetime
import logging
import sys


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

log_file = os.path.join(os.getcwd(), config['log_file'])
# Practise data
input_folder_path = config['input_folder_path']
# Ingesteddata
output_folder_path = config['output_folder_path']

control_file_path = os.path.join(output_folder_path, "ingestedfiles.txt")

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

#############Function for data ingestion directory
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file

    logging.info("[ DATA: Collecting data... ]")

    # Print directory listing
    def list_files(input_folder_path):
        for lists in os.listdir(input_folder_path): 
            path = os.path.join(input_folder_path, lists) 
            #print(path) 
            logging.info(f"[ DIRECTORY LISTING: {path} ]")
            if os.path.isdir(path): 
                list_files(path)
    
    list_files(input_folder_path)

    df = pd.DataFrame(
        columns=[
            "corporation", 
            "lastmonth_activity", 
            "lastyear_activity",
            "number_of_employees", 
            "exited"
        ]
    )

    def get_file_size(file_path):
        return os.path.getsize(file_path)

    all_data = pd.DataFrame()

    with open(control_file_path, "a+") as control_file:
        # For any given path look for subdirectories and collect filenames
        for root, _, files in os.walk(input_folder_path):
            #print(files)
            logging.info(f"[ FILE: Importing the following data files: {files} ]")
            for eachFileName in files:
                if eachFileName.endswith(".csv"):
                    file_path = os.path.join(root, eachFileName)
                    df = pd.read_csv(file_path)
                    all_data = pd.concat([all_data, df])
                    # Record filename, date/time of addition, and file size in the control file
                    file_size = get_file_size(file_path)
                    timestamp = datetime.now().replace(microsecond=0)
                    control_file.write(f"{timestamp}, FILE: {eachFileName}, SIZE: {file_size} bytes\n")
    
    logging.info(f"[ FILE: Data files information saved to '{output_folder_path}/ingestedfiles.txt' ]")

    # Remove duplicates
    allData = df.drop_duplicates()

    # Save to CSV
    allData.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)
    logging.info(f"[ FILE: Final data file saved to '{output_folder_path}/finaldata.csv' ]")

if __name__ == '__main__':
    merge_multiple_dataframe()
