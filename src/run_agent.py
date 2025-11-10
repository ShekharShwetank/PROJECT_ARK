from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import os
import re
import json

# Import all tools
from tools import (
    get_disk_usage,
    get_ram_usage,
    get_cpu_usage,
    list_directory,
    run_shell_command,
    read_file,
    get_directory_size,
    query_system_knowledge,
    query_project_knowledge
)

# --- CONFIGURATION ---
LLM_MODEL_NAME = "mistral-openorca"

# Create a mapping of tool names to functions
TOOLS_MAP = {
    "get_disk_usage": get_disk_usage,
    "get_ram_usage": get_ram_usage,
    "get_cpu_usage": get_cpu_usage,
    "list_directory": list_directory,
    "run_shell_command": run_shell_command,
    "read_file": read_file,
    "get_directory_size": get_directory_size,
    "query_system_knowledge": query_system_knowledge,
    "query_project_knowledge": query_project_knowledge,
}

def get_tool_descriptions():
    """Generate tool descriptions for the prompt."""
    descriptions = []
    for name, tool in TOOLS_MAP.items():
        descriptions.append(f"- {name}: {tool.description}")
    return "\n".join(descriptions)

def parse_thought_and_action(text):
    """
    Parse ONLY Thought and Action from the response.
    Ignore anything after Action Input (like fake Observations or Final Answers).
    """
    # Find the Thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|Final Answer:|$)', text, re.IGNORECASE | re.DOTALL)
    
    # Find Action and Action Input (but stop at any fake Observation or Final Answer)
    action_match = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
    input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
    
    # Check for a final answer
    final_answer_match = re.search(r'Final Answer:\s*(.*)', text, re.IGNORECASE | re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else ""
    
    if action_match:
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""
        action_input = action_input.strip('"\'[]')
        return thought, action, action_input, None
    
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
        return thought, None, None, final_answer

    # If no action or final answer, return the thought or the whole text
    return thought, None, None, text.strip()

def execute_tool(tool_name, tool_input):
    """Execute a tool and return its output."""
    if tool_name not in TOOLS_MAP:
        return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(TOOLS_MAP.keys())}"

    try:
        tool = TOOLS_MAP[tool_name]
        kwargs = {}

        # Prepare arguments based on the tool being called
        if tool_name in ["get_disk_usage", "get_ram_usage", "get_cpu_usage"]:
            pass # No arguments needed
        elif tool_name in ["list_directory", "get_directory_size", "read_file"]:
            from common import normalize_path
            kwargs["path"] = normalize_path(tool_input)
        elif tool_name == "run_shell_command":
            kwargs["command"] = tool_input
        else:
            kwargs["query"] = tool_input

        # Invoke the tool with the prepared arguments
        result = tool.invoke(kwargs)
        return result
    except Exception as e:
        return f"Error executing tool: {str(e)}"

def run_agent(llm, question, max_iterations=5, verbose=True):
    """
    Run a simplified ReAct agent with strict observation enforcement.
    """
    tool_descriptions = get_tool_descriptions()
    
    # Get current user and working directory for context
    current_user = os.getenv("USER", "unknown_user")
    current_dir = os.getcwd()
    
    prompt_template = """You are ARK, a helpful assistant. Answer the user's question.

System Context:
- Current User: {current_user}
- Current Directory: {current_dir}
Available Tools:
{tool_descriptions}

IMPORTANT RULES FOR FILE PATHS:
- When the user asks about personal files or general directories (like 'Documents', 'Desktop', 'Downloads'), ALWAYS assume the path starts from the user's home directory (`~`). For example, "my documents" should be interpreted as `~/Documents`.
- Do NOT assume the path is relative to the agent's current directory unless the user explicitly says so (e.g., "in the current directory").

INSTRUCTIONS:
1. **Think Step-by-Step**: Analyze the user's question and the history.
2. **Decide to Act or Answer**:
   - **For file tools (`list_directory`, `read_file`)**: Your `Action Input` MUST be a simple, human-like path (e.g., "application documents folder", "~/Desktop", "src/scripts"). Do NOT construct a full absolute path like `/home/ank/Desktop/...`. The system will normalize it for you.
   - **Handle Actionable Errors**: If the last `Observation` starts with `ACTION_REQUIRED`, it means a setup step is missing. Your next action MUST be to use the `run_shell_command` tool to execute the commands described in the error message. Run them one by one. After the commands succeed, you should then retry the original action that failed.
   - **IMPORTANT**: A "collection" is a database, NOT a file directory. You CANNOT create it with `mkdir`. You MUST use the `run_shell_command` tool to execute the provided scripts (`gather_system_info.sh`, `ingest.py`). Do NOT invent your own commands.
   - If you need more information and there is no actionable error, use the most appropriate tool. Respond with `Thought`, `Action`, `Action Input`.
   - **Path Recovery Strategy**: If a file or directory operation fails with an error like "No such file or directory" or "not found", DO NOT repeat the same command. Your next action MUST be to use `list_directory` on the parent directory to investigate the correct spelling or path. Analyze the output to find the correct name, then retry your original goal.
   - **For summarization requests**: If asked to summarize or explain files in a directory, your process MUST be: 1. Use `list_directory` to see the files. 2. Use the `read_file` tool on each relevant file in a loop. 3. Once you have the content, provide a `Final Answer` summarizing the information.
   - If you have enough information to answer, respond with `Thought` and `Final Answer`.
3. **Use one of the following formats**:

Format 1 (Use a tool):
Thought: [Your reasoning]
Action: [one of: {tool_names}]
Action Input: [The input for the tool]

Format 2 (Provide the final answer):
Thought: [Your reasoning for the final answer]
Final Answer: [The complete answer to the user's question]

Previous steps:
{history}

Question: {question}
Thought:"""

    tool_names = ", ".join(TOOLS_MAP.keys())
    history = ""
    
    # Check if it's a capability question first
    capability_keywords = ["what can you do", "who are you", "what are you", "your capabilities", "introduce yourself"]
    if any(kw in question.lower() for kw in capability_keywords):
        return "I am ARK, a helpful AI assistant. I can help you with:\n• Checking disk usage (get_disk_usage)\n• Checking RAM usage (get_ram_usage)\n• Checking CPU usage (get_cpu_usage)\n• Listing directory contents (list_directory)\n• Answering questions about your system configuration (query_system_knowledge)\n• Answering questions about your project code (query_project_knowledge)", []
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"🔄 Iteration {iteration + 1}")
            print(f"{'─'*70}")
        
        # Build prompt
        current_prompt = prompt_template.format(
            tool_descriptions=tool_descriptions,
            current_user=current_user,
            current_dir=current_dir,
            tool_names=tool_names,
            history=history,
            question=question
        )
        
        # Get LLM response
        response = llm.invoke(current_prompt)
        
        if verbose:
            print(f"\n💭 LLM Output:")
            print(response)
        
        # Parse thought and action (ignore everything else)
        thought, action_name, action_input, final_answer = parse_thought_and_action(response)
        
        if final_answer:
            if verbose:
                print("\n✅ Agent decided to provide a final answer.")
            return final_answer, [response]

        if not action_name:
            if verbose:
                print("\n⚠️ No action or final answer found. Returning last thought.")
            return thought or "I'm not sure how to proceed.", [response]
        
        if verbose:
            print(f"\n🔧 Executing Tool: {action_name}")
            print(f"📥 Input: {action_input if action_input else '(none)'}")
        
        # Execute the tool
        observation = execute_tool(action_name, action_input)
        
        if verbose:
            print(f"\n📤 Tool Output:")
            print(observation[:500] + "..." if len(observation) > 500 else observation)
        
        # Append the observation to the history for the next iteration
        history += f"""Thought: {thought}
Action: {action_name}
Action Input: {action_input}
Observation: {observation}\n"""
    
    return "I wasn't able to complete this task within the iteration limit.", []

def main():
    """
    Main function to run the Unified ARK Agent.
    """
    print("\n" + "="*70)
    print(" "*20 + "🤖 ARK AGENT v2.2")
    print("="*70)
    print("\nInitializing...")

    llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0)
    
    print(f"✓ LLM Model: {LLM_MODEL_NAME}")
    print(f"✓ Tools loaded: {', '.join(TOOLS_MAP.keys())}")
    print("✓ Path normalization: application documents folder -> ~/Documents/APPLICATION DOCUMENTS")
    print("\n" + "="*70)
    print("Ready! Type 'exit' or 'quit' to end the session.")
    print("Type 'verbose on/off' to toggle detailed reasoning display.")
    print("Type 'help' to see available commands.")
    print("\n💡 Example questions to try:")
    print("  • What's my current RAM usage?")
    print("  • What's the disk usage?")
    print("  • What is the current CPU usage?")
    print("  • What files are in ~/Desktop?")
    print("  • What files are in the application documents folder?")
    print("  • Explain the files in the 'scripts' folder.")
    print("  • Run 'tree ~/Documents' to see directory structure")
    print("  • What can you do?")
    print("="*70 + "\n")

    verbose = True

    while True:
        try:
            user_input = input("💬 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if user_input.lower() == 'help':
                print("\n📚 Available Commands:")
                print("  - 'verbose on/off' - Toggle detailed reasoning")
                print("  - 'help' - Show this help message")
                print("  - 'exit/quit' - End the session")
                print("\n🔧 Available Tools:")
                for name, tool in TOOLS_MAP.items():
                    print(f"  - {name}")
                print()
                continue
            
            if user_input.lower().startswith('verbose'):
                if 'off' in user_input.lower():
                    verbose = False
                    print("✓ Verbose mode OFF - showing only final answers\n")
                else:
                    verbose = True
                    print("✓ Verbose mode ON - showing full reasoning\n")
                continue
            
            if not user_input:
                continue
            
            # Run the agent
            answer, reasoning = run_agent(llm, user_input, verbose=verbose)
            
            if not verbose:
                print(f"\n🤖 ARK: {answer}\n")
            else:
                print("\n" + "="*70)
                print("✨ FINAL ANSWER")
                print("="*70)
                print(f"\n🤖 ARK: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("👋 ARK Agent shutting down. Goodbye!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()