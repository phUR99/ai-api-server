from typing import Union
import model
from fastapi import FastAPI

modelAnd = model.AndModel()
modelXor = model.XorModel()
modelOr = model.OrModel()
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}") #endpoint
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

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
        return {"error" : f"error is occured{str(e)}"}

@app.post("/train{model}")
def train(model:str):
    try:
        if model == 'And':
            modelAnd.train()    
        elif model == 'Xor':
            modelOr.train()
        elif model == 'Or':
            modelXor.train()
    except Exception as e:
        return {"error" : f"error is occured{str(e)}"}    
    return {"result" : "OK"}
   
