import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist
import yaml

app = FastAPI()

PREDICTION_COLS = ['date','store','item']


#Creating a class for the attributes input to the ML model.
# class water_metrics(BaseModel):
#     ph : float
#     Hardness :float
#     Solids : float
#     Chloramines : float
#     Sulfate : float
#     Conductivity : float
#     Organic_carbon : float
#     Trihalomethanes : float
#     Turbidity : float

# Represents a datapoint for the attributes input to the ML model.
class Data(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]

#Loading the trained model
with open("./finalized_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


@app.post('/prediction' )
def get_potability(data: water_metrics):
    received = data.dict()
    ph = received['ph']
    Hardness = received['Hardness']
    Solids = received['Solids']
    Chloramines = received['Chloramines']
    Sulfate = received['Sulfate']
    Conductivity = received['Conductivity']
    Organic_carbon = received['Organic_carbon']
    Trihalomethanes = received['Trihalomethanes']
    Turbidity = received['Turbidity']
    pred_name = loaded_model.predict([[ph, Hardness, Solids,
                                       Chloramines, Sulfate, Conductivity, Organic_carbon,
                                       Trihalomethanes,Turbidity]]).tolist()[0]
    return {'Prediction':  pred_name}