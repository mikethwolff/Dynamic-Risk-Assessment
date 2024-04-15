import requests
import json
import os

#Specify a URL that resolves to your workspace
#URL = "http://localhost/"
URL = "http://127.0.0.1:8000/"


#Call each API endpoint and store the responses
response1 = requests.post(f"{URL}/prediction", json={"filepath": "testdata.csv"}).text
response2 = requests.get(f"{URL}/scoring").text
response3 = requests.get(f"{URL}/summarystats").text
response4 = requests.get(f"{URL}/diagnostics/timing").text
#curl http://localhost:8000/diagnostics/timing

#combine all API responses
#responses = #combine reponses here
responses = response1 + "\n" + response2 + "\n" + response3 + "\n" + response4

#write the responses to your workspace
with open('config.json', 'r') as file:
    config = json.load(file)
    model_path = os.path.join(config['output_model_path'])

with open(os.path.join(model_path, "apireturns.txt"), "w") as file:
    file.write(responses)
