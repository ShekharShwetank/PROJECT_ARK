from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
# Import all three of our tools
from src.tools import get_disk_usage, query_system_knowledge, query_project_knowledge

# --- CONFIGURATION ---
LLM_MODEL_NAME = "mistral-openorca"

def main():
    """
    Main function to run the Unified ARK Agent.
    """
    print("--- Initializing Unified ARK Agent ---")

    # --- 1. Initialize the LLM and list all available tools ---
    llm = OllamaLLM(model=LLM_MODEL_NAME)

    tools = [
        get_disk_usage,
        query_system_knowledge,
        query_project_knowledge,
    ]

    print(f"Tools loaded: {[tool.name for tool in tools]}")

    # --- 2. Create the Agent Prompt ---
    # The ReAct prompt template remains the same, it automatically incorporates the tool descriptions.
    react_prompt_template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(react_prompt_template)

    # --- 3. Create the Agent and Executor ---
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True 
    )
    print("Agent Executor created.")

    print("\n--- Unified ARK Agent is ready. ---")

    # --- 4. Interactive Query Loop ---
    while True:
        try:
            user_question = input("\n> You: ")
            if user_question.lower() in ['exit', 'quit']:
                break

            response = agent_executor.invoke({"input": user_question})

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

    print("\nARK Agent shutting down. Goodbye.")

if __name__ == "__main__":
    main()
