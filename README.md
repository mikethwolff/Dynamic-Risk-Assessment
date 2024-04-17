# Machine Learning Model Risk Assessment

## Project starter kit

ML DevOps project: Create, deploy, and monitor a risk assessment machine learning model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Data has been downloaded from the [Udacity 60412fe6 project starter kit](https://video.udacity-data.com/topher/2021/March/60412fe6_starter-file/starter-file.zip).


## Environment

- Create your conda environment: 
  ```
  $ conda create --name <your environment name> --file requirements.txt
  $ conda env create --file conda.yaml
  
  $ conda activate <your environment name>
  ```

NOTE: If you received the following error message: "Failed to build numpy Pillow scikit-learn scipy" execute the following:

```
$ conda install scikit-learn pillow
```


## Install Jupiter notebook:

```
$ conda install jupyter
```
The Jupyter notebook ["ModelScoring_Data.ipynb"](ModelScoring_Data.ipynb) will give you a good overview of the starter data.


## Project files and purpose:

You can run each project file as follows:

```bash
# Ingest data
python ingestion.py
# Train model
python training.py
# Deploy model
python deployment.py
# Score model
python scoring.py
# Report
python reporting.py
# Run diagnostics
python diagnostics.py
```
You can run each file on its own. Alternatively, run the full process as follows:

NOTE: Remember to first start the API on the local server - on another console - as mentioned below.

```bash
python full_process.py
```
The full_process will also check the ingested data and re-train the model if necessary.

## Logs and metrics

The file ["practicemodels/apireturns.txt"](/practicemodels/apireturns.txt) contains the metrics of the model with the ingested trainings data,
while the file ["models/apireturns.txt"](/models/apireturns.txt) the metrics of the model with the new ingested data.

The file ["logs/churn.log"](/logs/churn.log) collects general data during the process.

## Local server

If we want to start the diagnosing API, execute the following:

```bash
# Console 1: Start the API on the web server: http://localhost:8000
python app.py
# Console 2: Perform API access calls
python api_calls.py
```
