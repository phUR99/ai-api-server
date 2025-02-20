from typing import Union
import model
import os
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
            model_path = "and_model.pth"
            if os.path.exists(model_path):
                modelAnd.load(model_path)
            else:
                modelAnd.train()
                modelAnd.save(model_path)
            result = modelAnd.predict([left, right])
        elif model == 'Xor':
            model_path = "xor_model.pth"
            if os.path.exists(model_path):
                modelXor.load(model_path)
            else:
                modelXor.train()
                modelXor.save(model_path)
            result = modelXor.predict([left, right])
        elif model == 'Or':
            model_path = "or_model.pth"
            if os.path.exists(model_path):
                modelOr.load(model_path)
            else:
                modelOr.train()
                modelOr.save(model_path)
            result = modelOr.predict([left, right])
        return {"result": result}
    except Exception as e:
        return {"error" : f"error is occurred{str(e)}"}

@app.post("/train{model}")
def train(model:str):
    try:
        if model == 'And':
            modelAnd.train()
            modelAnd.save("and_model.pth")
        elif model == 'Xor':
            modelXor.train()
            modelXor.save("xor_model.pth")
        elif model == 'Or':
            modelOr.train()
            modelOr.save("or_model.pth")
    except Exception as e:
        return {"error" : f"error is occurred{str(e)}"}    
    return {"result" : "OK"}

