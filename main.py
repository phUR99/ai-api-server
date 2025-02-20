from typing import Union
import model
from fastapi import FastAPI
import gdown
import os
_id= {
    'and_file' : "1PsqkwTk0-7N8Jn-f4OuQamDnc5-roYJp",
    'not_file' : "1OaoWXrKxq10viz3e3omDgEpiehogaukU",
    'xor_file' : "1RnBZD0WtXM3YXc1LwmlXpymb9TszuW-v",
    'or_file'  : "1OG6ABmD185Z9Kk74iXSpiRgOBHcnwLWY"
}
modelAnd = model.AndModel()
if(not os.path.isfile('and.pt')):
    gdown.download(id=_id['and_file'], output="and.pt", quiet=False)   
    modelAnd.load('and.pt')
modelXor = model.XorModel()
if(not os.path.isfile('xor.pt')):
    gdown.download(id=_id['xor_file'], output="xor.pt", quiet=False)   
    modelXor.load('xor.pt')
modelOr = model.OrModel()
if(not os.path.isfile('or.pt')):
    gdown.download(id=_id['or_file'], output="or.pt", quiet=False)   
    modelAnd.load('or.pt')
modelNot = model.NotModel()

if(not os.path.isfile('not.pt')):
    gdown.download(id=_id['not_file'], output="not.pt", quiet=False)   
    modelAnd.load('not.pt')

app = FastAPI()

@app.get("/predict{model}/left/{left}/right/{right}") #endpoint
def predict(model : str, left: int, right: int):
    try:
        if model == 'And':
            result = modelAnd.predict([left, right])
        elif model == 'Xor':
            result = modelXor.predict([left, right])
        elif model == 'Or':
            result = modelOr.predict([left, right])
        elif model == 'Not':
            result = modelNot.predict([left])
        return {"result": result}
    except Exception as e:
        return {"error" : f"error is occurred{str(e)}"}

@app.post("/train{model}")
def train(model:str):
    try:
        if model == 'And':
            modelAnd.train()    
        elif model == 'Xor':
            modelXor.train()
        elif model == 'Or':
            modelOr.train()
        elif model == 'Not':
            modelNot.train()            
    except Exception as e:
        return {"error" : f"error is occurred{str(e)}"}    
    return {"result" : "OK"}
   
