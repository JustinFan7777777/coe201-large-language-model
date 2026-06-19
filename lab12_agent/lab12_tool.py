import inspect
import json

def parse_signature(func):
    """
    (Provided) Helper to extract properties and required fields from a function signature.
    """
    sig = inspect.signature(func)
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        param_type = "string"
        if param.annotation == int:
            param_type = "integer"
        elif param.annotation == float:
            param_type = "number"
        elif param.annotation == bool:
            param_type = "boolean"
            
        properties[name] = {"type": param_type}
        if param.default == inspect.Parameter.empty:
            required.append(name)
            
    return properties, required


def my_tool(func):
    """
    A custom decorator that converts a python function into an OpenAI tool schema.
    The schema should be stored in `func.args_schema`.
    
    Example Schema for `get_weather(city: str, days: int = 1)`:
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string" },
            "days": { "type": "integer" }
          },
          "required": ["city"]
        }
      }
    }
    """
    # 1. Use the provided helper to get properties and required arguments
    properties, required = parse_signature(func)
    
    # 2. Extract the function name and docstring (description)
    ### TODO: Extract func.__name__ and func.__doc__
    name = func.__name__
    description = (func.__doc__ or "").strip()
    
    # 3. Build the OpenAI-compatible JSON schema dictionary
    ### TODO: Construct the schema dict with "type": "function" and "function": {...}
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
            }
        }
    }
    
    
    # 4. Attach the schema to the function
    ### TODO: Set func.args_schema = schema
    if required:
        schema["function"]["parameters"]["required"] = required

    func.args_schema = schema
    
    return func


if __name__ == "__main__":
    @my_tool
    def get_weather(city: str, days: int = 1) -> str:
        """Get the current weather for a city."""
        pass
    
    # 1. Print the generated schema
    print("--- Generated Schema ---")
    print(json.dumps(getattr(get_weather, "args_schema", {}), indent=2))
    
    # 2. Test if the LLM accepts this schema
    print("\n--- Testing API Tool Call ---")
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm = ChatOpenAI(
        base_url="http://172.18.36.182:8000/v1",
        api_key="sk-coe201-qwen-agent",
        model="qwen36-agent-27b",
        temperature=0
    )
    
    try:
        # We bind the raw schema dictionary directly
        llm_with_tools = llm.bind_tools([getattr(get_weather, "args_schema", {})])
        msg = llm_with_tools.invoke([HumanMessage(content="What's the weather in Shenzhen?")])
        
        if getattr(msg, "tool_calls", None):
            print("Success! The model called the tool:")
            for call in msg.tool_calls:
                print(f"- {call['name']}: {call['args']}")
        else:
            print("Failure. The model did not call any tools.")
            print(f"Model output: {msg.content}")
    except Exception as e:
        print(f"API Error: The schema might be invalid. Error details:\n{e}")
