import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# Initialize the ChatModel
llm = ChatOpenAI(
    base_url="http://172.18.36.182:8000/v1",
    api_key="sk-coe201-qwen-agent",
    model="qwen36-agent-27b",
    temperature=0.1
)

@tool
def get_weather(thought: str, city: str) -> str:
    """Get the current weather for a city. Use the thought parameter to explain your reasoning."""
    return "32C" if city == "Shenzhen" else "25C"

@tool
def convert_currency(thought: str, amount: float, from_cur: str, to_cur: str) -> str:
    """Convert currency amount. Use the thought parameter to explain your reasoning."""
    rate = 7.24 if from_cur == "USD" else 1.0
    return f"Converted: {amount * rate:.2f}"

tools = [get_weather, convert_currency]
tool_map = {t.name: t for t in tools}

# Bind tools to the model
### TODO: Bind the tools to the LLM (for LangChain tools, you can just pass the list `tools` to bind_tools)
llm_with_tools = llm.bind_tools(tools)

def run_agent_loop(prompt: str):
    """
    Run the ReAct agent loop.
    REQUIREMENT: You must print out the [Thought], [Tool Call], and [Observation] at each step
    so that the agent's internal reasoning process is visible in the console.
    """
    messages = [
        SystemMessage(content="Use tools when needed. Explain your thought process in the 'thought' parameter of every tool you call."),
        HumanMessage(content=prompt)
    ]
    
    for step in range(10):
        print(f"\n--- Step {step+1} ---")
        
        # 1. Invoke the model
        ### TODO: Invoke llm_with_tools using the `messages` list, and append the result ai_msg back to `messages`
        ai_msg = llm_with_tools.invoke(messages)

        messages.append(ai_msg)
        
        # Optional: Print the thought process if the model outputs any content
        if ai_msg and ai_msg.content and ai_msg.content.strip():
            print(f"[Thought]: {ai_msg.content.strip()}")
        
        # 2. Check if the model wants to call any tools
        ### TODO: Break the loop if ai_msg.tool_calls is empty
        if not getattr(ai_msg, "tool_calls", None):
            break
            
        # 3. Execute tool calls and append results
        ### TODO: Iterate over ai_msg.tool_calls
        ###       REQUIREMENT: For each call, you must print the following in order:
        ###       1. `[Thought]: <thought_content>` (extract from call["args"]["thought"] if present)
        ###       2. `[Tool Call]: <function_name>(<args>)`
        ###       3. Execute the python function from `tool_map` using `.invoke(call["args"])` and print `[Observation]: <result>`
        ###       Finally, append the result back to messages as a ToolMessage using call["id"]

        for call in ai_msg.tool_calls:
            call_args = dict(call["args"])

            thought_content = call_args.get("thought", "")
            if thought_content:
                print(f"[Thought]: {thought_content}")

            display_args = dict(call_args)
            display_args.pop("thought", None)

            print(f"[Tool Call]: {call['name']}({display_args})")

            tool_result = tool_map[call["name"]].invoke(call_args)
            print(f"[Observation]: {tool_result}")

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=call["id"]
                )
            )

    return messages[-1].content

if __name__ == "__main__":
    answer = run_agent_loop("Check Shenzhen weather. If > 30C, convert 50 USD to CNY; else 100 USD.")
    print("\n[Final Answer]:", answer)
