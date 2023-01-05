# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020
@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

from pydantic import BaseModel

import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
import json



# 2. Create the app object
app = FastAPI()

# Import model
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

# Import data
data_test = open("data_test.pkl","rb")
data_test = pickle.load(data_test)

data_train = open("data_train.pkl","rb")
data_train = pickle.load(data_train)

explainer = lime_tabular.LimeTabularExplainer(
    training_data = np.array(data_train.drop(['SK_ID_CURR'], axis =1)),
    feature_names = data_train.drop(['SK_ID_CURR'], axis =1).columns,
    class_names = [0, 1],
    mode = 'classification'
)



#  http://127.0.0.1:8000
@app.get('/')
def index():
    """
    message de base de l'API (non utilise par la suite)
    """
    return {'message': 'Bienvenue sur la prédiction du remboursement d un client Home Credit'}

# 3. Expose the prediction functionality

class Individu(BaseModel):
    id_cli:int

@app.post('/show')
def Print_individu(index_test:Individu):
    """ 
    Retourne la ligne avec les informations concernant l'individu teste
    """
    index_ = index_test.id_cli
  #  print(index_)
    individu_teste = data_test[data_test['SK_ID_CURR'] == index_].drop(['SK_ID_CURR'], axis =1).to_json(orient = 'records')

    return individu_teste

@app.post('/predict')
def predict_defaut(index_test:Individu):
    """
    Renvoi la prediction du modele pour l'individu teste
    """
    index_ = index_test.id_cli
    #print(index_)
    individu_teste = data_test[data_test['SK_ID_CURR'] == index_].drop(['SK_ID_CURR'], axis =1)
   # print(individu_teste)
 #   print(classifier.predict(individu_teste))
    prediction = classifier.predict_proba(individu_teste)[0][1]

    return {
        'prediction': prediction
    }

@app.post('/explain')
def explain_feature(index_test:Individu):
    """
    Renvoi la prediction par LIME ainsi que l'explication de la decision pour l'individu teste
    """
    index_ = index_test.id_cli
    #print(index_)
    individu_teste = data_test[data_test['SK_ID_CURR'] == index_].drop(['SK_ID_CURR'], axis =1).values
   # print(individu_teste)
 #   print(classifier.predict(individu_teste))
    exp = explainer.explain_instance(
        data_row = individu_teste[0], 
        predict_fn = classifier.predict_proba
    )
    return json.dumps(exp.as_list())
        
    
    
# 5. Run the API with uvicorn    uvicorn app:app --reload
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    
    
    