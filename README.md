## Integration of a Mathematical Calulations with a Chat Completion System using LLM Function-Calling

### AIM:
To design and implement a Python function for calculating the volume of a cylinder, integrate it with a chat completion system utilizing the function-calling feature of a large language model (LLM).

### PROBLEM STATEMENT:
Large Language Models (LLMs) cannot inherently perform precise mathematical calculations like finding the volume of a geometry because they are language prediction systems rather than calculation engines. This project demonstrates how to overcome this limitation by using OpenAI's Function-Calling mechanism to securely delegate execution to a precise Python function that calculates a cylinder's volume (\(V = \pi r^2 h\)).

### DESIGN STEPS:
1. **Define the Core Logic:** Create a native Python function `calculate_cylinder_volume(radius, height)` that utilizes the `math` library to execute precise volume geometry operations.
2. **Construct JSON Schema:** Build a structural metadata payload defining the function name, description, parameters (`radius` and `height` as float numbers), and required parameters.
3. **Initialize Chat State:** Pass the tool specification schema along with the user's conversational prompt (e.g., "Find the volume of a cylinder with radius 5 and height 10") to the Chat Completions endpoint.
4. **Intercept Tool Arguments:** Extract the arguments payload generated programmatically by the model (`args["radius"]`, `args["height"]`), parse them using `json.loads()`, and run the local calculation.
5. **Generate Final Context:** Append the mathematical execution results back to the conversation thread using the `function` role, and call the model once more to create a friendly response.

### PROGRAM:
```python
import os
import openai
import json
import math
from dotenv import load_dotenv, find_dotenv

# Initialize client configuration
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# 1. Define the actual mathematical tool
def calculate_cylinder_volume(radius, height):
    """Calculates the volume of a cylinder given radius and height"""
    volume = math.pi * (float(radius) ** 2) * float(height)
    result_info = {
        "radius": radius,
        "height": height,
        "volume": round(volume, 2)
    }
    return json.dumps(result_info)

# 2. Define the JSON schema blueprint for the tool
functions = [
    {
        "name": "calculate_cylinder_volume",
        "description": "Calculate the volume of a cylinder given its radius and height",
        "parameters": {
            "type": "object",
            "properties": {
                "radius": {
                    "type": "number",
                    "description": "The radius of the circular base of the cylinder"
                },
                "height": {
                    "type": "number",
                    "description": "The vertical height of the cylinder"
                }
            },
            "required": ["radius", "height"]
        }
    }
]

# 3. Setup conversation history
messages = [
    {
        "role": "user",
        "content": "What is the volume of a cylinder with a radius of 3cm and a height of 5cm?"
    }
]

# 4. First endpoint request to interpret parameters
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=messages,
    functions=functions,
    function_call={"name": "calculate_cylinder_volume"}
)

response_message = response["choices"][0]["message"]
messages.append(response_message)

# 5. Extract arguments and run mathematical tool locally
args = json.loads(response_message["function_call"]["arguments"])
observation = calculate_cylinder_volume(radius=args["radius"], height=args["height"])

# 6. Feed context back to system
messages.append(
    {
        "role": "function",
        "name": "calculate_cylinder_volume",
        "content": observation,
    }
)

# 7. Final completion response generation
final_response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=messages,
)

print(final_response["choices"][0]["message"]["content"])
```

### OUTPUT:
<img width="1920" height="1080" alt="Screenshot (3010)" src="https://github.com/user-attachments/assets/987a6522-7c0b-4d03-9a98-56c86ae99d5c" />


### RESULT:
The system successfully integrated a precise backend calculation workflow with an LLM chat completion flow. By exploiting function calling, the model dynamically verified user inputs, structured calculation inputs correctly into an executable state, run the arithmetic local routines without hallucinations, and formatted the data into a fluent conversational output.
