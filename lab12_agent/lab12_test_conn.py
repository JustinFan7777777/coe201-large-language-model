import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# 1. Initialize the ChatModel pointing to the remote server
# Replace with the actual URL and model name provided by the instructor
llm = ChatOpenAI(
    base_url="http://172.18.36.182:8000/v1",
    api_key="sk-coe201-qwen-agent",
    model="qwen36-agent-27b",
    temperature=0
)

# 2. Define a simple mock tool
@tool
def get_current_time(location: str) -> str:
    """Get the current time for a specific location."""
    return "12:00 PM"

# 3. Bind the tool to the model
### TODO: Use llm.bind_tools to bind the get_current_time tool
llm_with_tools = llm.bind_tools([get_current_time])

if __name__ == "__main__":
    prompt = "What time is it in New York?"
    print(f"User: {prompt}")
    
    # 4. Invoke the model
    ### TODO: Invoke llm_with_tools with a HumanMessage containing the prompt
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])

    print("\n--- Model Response ---")
    if response and getattr(response, "tool_calls", None):
        print("Success! The model decided to use a tool:")
        for call in response.tool_calls:
            print(f"- Tool Name: {call['name']}")
            print(f"- Tool Args: {call['args']}")
    else:
        print("Failure. The model did not return any tool calls.")
        print("Response content:", getattr(response, "content", "No response"))
