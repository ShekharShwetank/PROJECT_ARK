# src/run_agent.py

from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
import json
import sys
import os

# Import all tools
from src.tools import (
    get_system_spec,
    get_disk_usage,
    list_files,
    read_file_content,
    list_running_processes,
    create_file,
    create_kicad_project
)

def main():
    """Main function to run the Unified ARK Agent."""
    print("--- Initializing ARK Agent ---")

    # Initialize LLM - Using Mistral (verified working)
    llm = OllamaLLM(model="mistral", temperature=0.0)

    # Define tools
    tools = [
        get_system_spec,
        get_disk_usage,
        list_files,
        read_file_content,
        list_running_processes,
        create_file,
        create_kicad_project
    ]
    
    print(f"Tools loaded: {[tool.name for tool in tools]}")

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are ARK, an AI assistant for Ubuntu systems. "
            "When asked for system information, ALWAYS use the get_system_spec tool first.\n"
            "Respond in this EXACT JSON format:\n"
            "{\n"
            "  \"answer\": \"<your answer>\",\n"
            "  \"sources\": [\"<tool1>\", \"<tool2>\"],\n"
            "  \"action_required\": <true/false>,\n"
            "  \"next_steps\": [\"<step1>\", \"<step2>\"]\n"
            "}\n"
            "For CPU information, use get_system_spec with ['cpu_model', 'cpu_cores']"
        )),
        ("human", "{input}"),
    ])
    
    # Create the agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Prevents infinite loops
        prompt=prompt,
        early_stopping_method="generate"
    )
    
    print("ARK Agent initialized successfully.")
    print("\n--- ARK is ready. Type 'exit' to quit ---")

    while True:
        try:
            user_question = input("\n> You: ").strip()
            if user_question.lower() in ['exit', 'quit']:
                break
                
            if not user_question:
                continue
                
            # Invoke the agent
            response = agent.invoke({"input": user_question})
            
            # Format and display the response
            print("\n> ARK:")
            try:
                output = response["output"]
                if isinstance(output, str):
                    output = json.loads(output)
                
                print(f"Answer: {output.get('answer', 'No answer provided')}")
                if output.get("sources"):
                    print(f"Sources: {', '.join(output['sources'])}")
                if output.get("next_steps"):
                    print("Next Steps:")
                    for step in output["next_steps"]:
                        print(f"- {step}")
            except Exception as e:
                print(output)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

    print("\nARK Agent shutting down. Goodbye.")

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()