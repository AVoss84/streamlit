import json, requests

# see create_api.py for details

selected_op = 'Multiplication' 
x = 2
y = 4

inputs = {"operation": selected_op, "x": x, "y": y}

res = requests.post(url="http://127.0.0.1:8000/calculate", data = json.dumps(inputs))

print(res.text)