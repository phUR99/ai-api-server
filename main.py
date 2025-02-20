from typing import Union
import model
from fastapi import FastAPI

model = model.AndModel()
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}") #endpoint
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/left/{left}/right/{right}") #endpoint
def predict(left: int, right: int):
    result = model.predict([left, right])
    return {"result": result}

@app.post("/train")
def train():
    model.train()
    return {"result" : "OK"}
