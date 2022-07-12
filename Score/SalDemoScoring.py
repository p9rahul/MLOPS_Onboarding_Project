import imp
import json
from operator import mod
from pyexpat import model
from statistics import mode
import numpy as np
import os
import pickle
from sklearn,externals import joblib
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model

"""
The init method is called once, when the web service starts up
#Typically you would deserialize the model file , as shown here using joblib 
# and store it in a global variable so your run() method can access it later.
"""

def init():
    global model
     #The AZUREML_MODEL_DIR enviroment variable indicates a directory containing the model file you registered.
    model_filename ='sal_model.pkl'
    model_path =os.path.join(os.environ['AZUREML_MODEL_DIR'],model_filename )

    model = joblib.load(model_path)

#The run() method is called each time a request is made to the scoring API
#shown here are the optional input_schema and output_schema decorators
#from the inference-schema pip package. using these decorators on your
#run() method parses and validates the incoming payload against 
#the example input you provide here. This will aslo generate a swagger API document for your web service

def run(data):
 data =np.array(json.loads(data)['data'])

 #make prediction 
 y_hat = model.predict(data)
 return json.dumps(y_hat.tolist())