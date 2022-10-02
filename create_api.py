
from unittest import result
from fastapi import FastAPI
from pydantic import BaseModel

from src.test_fct_fastapi import calculate

#calculate(x = 2, y = 3, operation= 'Multiplication')

class User_input(BaseModel):
    operation : str
    x: float
    y: float


app = FastAPI(title="Test API", description="Alex API", version="1.0")

@app.get("/health")
async def root():
    return {"message": "Service is healthy!"}


@app.post("/calculate")
def operate(input:User_input):
    result = calculate(input.operation, input.x, input.y)
    return result


# Start web server
# uvicorn create_api:app --reload