import os
import subprocess
import shutil
import traceback
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich.markdown import Markdown

console = Console()

# Initialize ChatModel
llm = ChatOpenAI(
    base_url="http://172.18.36.182:8000/v1",
    api_key="sk-coe201-qwen-agent",
    model="qwen36-agent-27b",
    temperature=0
)

# 1. Define the Python Sandbox Tool
def is_safe_code(code: str) -> bool:
    """Use LLM to judge if python code is safe."""
    ### TODO: Write a prompt instructing the LLM to judge if the python code is safe or not.
    ### HINT: Specify rules for safe/unsafe actions (e.g., creating/moving files is safe, 
    ###       but removing/deleting files or using subprocess is unsafe). 
    ###       Ask the LLM to output exactly 'YES' or 'NO'.
    prompt = f"""
You are a security classifier for Python code.

Decide whether the following Python code is safe to execute in a local sandbox.

Safe code:
- Reading or listing files and folders
- Creating folders
- Moving or renaming files inside the workspace
- Simple string / list / dictionary manipulation
- Printing output

Unsafe code:
- Deleting files or folders
- Overwriting important files
- Using subprocess
- Using os.system, shutil.rmtree, os.remove, os.unlink, os.rmdir, pathlib.Path.unlink
- Network access
- Infinite loops or code that tries to escape the sandbox
- Any destructive system operation

Return exactly one word:
YES if the code is safe.
NO if the code is unsafe.

Code:
{code}
""".strip()
    
    ### TODO: Invoke the LLM with the prompt using HumanMessage. 
    response = llm.invoke([HumanMessage(content=prompt)])
    
    ### TODO: Return True if the LLM's response contains 'YES', False otherwise.
    return "YES" in str(getattr(response, "content", "")).upper()
@tool
def execute_python(thought: str, code: str) -> str:
    """Executes python code and returns the printed output or error. Use the thought parameter to explain your reasoning."""
    import sys
    from io import StringIO
    
    ### TODO: 

    ### 1. Call is_safe_code(code). If it returns False, return a string indicating a Security Error.
    if not is_safe_code(code):
        return "Security Error: unsafe code rejected by sandbox."

    ### 2. Redirect sys.stdout to a StringIO object to capture the printed output of the code.
    captured_output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output

    ### 3. Use a try-except block to execute the code using `exec(code, {"os": os, "shutil": shutil, "print": print})`.
    try:
        exec(code, {"os": os, "shutil": shutil, "print": print})
        output = captured_output.getvalue().strip()

        ### 4. If execution is successful, return the captured output (if empty, return "Success").
        return output if output else "Success"
    
    ### 5. If an exception occurs, return a string containing the exception type and message.
    except Exception:
        return traceback.format_exc()

    ### 6. IMPORTANT: Always restore sys.stdout in a `finally` block to prevent printing issues.
    finally:
        sys.stdout = original_stdout

### TODO: Define a list containing the execute_python tool
tools = [execute_python]

### TODO: Create a dictionary mapping the tool's __name__ to the tool function
tool_map = {execute_python.name: execute_python}

### TODO: Bind the tools to the LLM. 
### HINT: Since we use LangChain's `@tool`, you can just pass the list `tools` to `bind_tools`.
llm_with_tools = llm.bind_tools(tools)

def setup_test_folder():
    """Helper to create a messy folder for the agent to organize."""
    if os.path.exists("test_folder"):
        shutil.rmtree("test_folder")
    os.makedirs("test_folder")
    open("test_folder/1.png", "w").close()
    open("test_folder/2.png", "w").close()
    open("test_folder/doc.txt", "w").close()

def print_directory_structure(path="test_folder"):
    """Helper to print the directory structure using rich.tree.Tree."""
    tree = Tree(f"📁 [bold]{path}[/bold]")
    
    def add_children(dir_path, tree_node):
        try:
            entries = sorted(os.listdir(dir_path))
        except Exception:
            return
            
        for entry in entries:
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                branch = tree_node.add(f"📁 [bold]{entry}[/bold]")
                add_children(full_path, branch)
            else:
                tree_node.add(f"📄 {entry}")
                
    add_children(path, tree)
    console.print()
    console.print(tree)
    console.print()

def run_agent_loop(prompt: str):
    messages = [
        SystemMessage(content="You are a helpful system administrator. Use tools to complete your tasks."),
        HumanMessage(content=prompt)
    ]
    
    for step in range(10):
        console.print(f"\n[bold cyan]--- Step {step+1} ---[/bold cyan]")
        
        # Invoke LLM
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        if getattr(ai_msg, "content", None) and ai_msg.content.strip():
            console.print(Panel(Markdown(ai_msg.content.strip()), title="[bold blue]Thought[/bold blue]", border_style="blue"))
        
        if not getattr(ai_msg, "tool_calls", None):
            console.print("[bold green]Final Answer Reached![/bold green]")
            break
            
        # Execute tools
        for call in ai_msg.tool_calls:
            thought = call["args"].get("thought", "")
            if thought and thought.strip():
                console.print(Panel(Markdown(thought.strip()), title="[bold blue]Thought[/bold blue]", border_style="blue"))
                
            code = call['args'].get('code', '')
            if code:
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"[bold magenta]Tool Call: {call['name']}[/bold magenta]", border_style="magenta"))
            else:
                display_args = {k: v for k, v in call["args"].items() if k != "thought"}
                console.print(Panel(str(display_args), title=f"[bold magenta]Tool Call: {call['name']}[/bold magenta]", border_style="magenta"))
            
            tool_func = tool_map.get(call["name"])
            if tool_func:
                result = tool_func.invoke(call["args"])
            else:
                result = "Tool not found."
                
            console.print(Panel(str(result).strip(), title="[bold yellow]Observation[/bold yellow]", border_style="yellow"))
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return getattr(messages[-1], "content", "")

if __name__ == "__main__":
    setup_test_folder()
    console.print("[bold green]Created test_folder with .png and .txt files.[/bold green]")
    print_directory_structure()
    
    ### TODO: Write a prompt that instructs the agent to organize the 'test_folder' directory.
    ### REQUIREMENT: 
    ### 1. Force the agent to FIRST try to import a non-existent module 'organize_helper' 
    ###    and run `organize_helper.organize_all()`.
    ### 2. Instruct the agent to NOT use try-except blocks (so the tool directly returns the error to the agent).
    ### 3. Tell the agent that if it encounters an ImportError, it must self-correct by using 
    ###    'os' and 'shutil' to move all '.png' files into 'test_folder/images'.
    prompt = """
Organize the folder named test_folder.

First, try to import a module named organize_helper and run organize_helper.organize_all().
Do not use try-except blocks.

If you receive an ImportError, self-correct by using os and shutil to move all .png files from test_folder into test_folder/images.
Create the images directory if it does not exist.
Only move the .png files.
"""
    
    answer = run_agent_loop(prompt)
    console.print(Panel(Markdown(answer), title="[bold green]Agent Finished[/bold green]", border_style="green"))
    
    print_directory_structure()
    
    # Verify
    if os.path.exists("test_folder/images/1.png"):
        console.print("[bold green]SUCCESS: The folder was successfully organized![/bold green]")
    else:
        console.print("[bold red]FAILURE: The folder was not organized correctly.[/bold red]")
