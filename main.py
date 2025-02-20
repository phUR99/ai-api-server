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

@app.get("/predictAnd/left/{left}/right/{right}") #endpoint
def predict(left: int, right: int):
    result = modelAnd.predict([left, right])
    return {"result": result}

@app.get("/predictOr/left/{left}/right/{right}") #endpoint
def predict(left: int, right: int):
    result = modelOr.predict([left, right])
    return {"result": result}

@app.get("/predictXor/left/{left}/right/{right}") #endpoint
def predict(left: int, right: int):
    result = modelXor.predict([left, right])
    return {"result": result}

@app.post("/trainAnd")
def train():
    modelAnd.train()    
    return {"result" : "OK"}
@app.post("/trainOr")
def train():
    modelOr.train()
    return {"result" : "OK"}
@app.post("/trainXor")
def train():
    modelXor.train()
    return {"result" : "OK"}    
