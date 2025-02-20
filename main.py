from typing import Union
import model
from fastapi import FastAPI

modelAnd = model.AndModel()
modelXor = model.XorModel()
modelOr = model.OrModel()
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
    except Exception as e:
        return {"error" : f"error is occurred{str(e)}"}    
    return {"result" : "OK"}
   
