import argparse
from rag_utils import _rag_utils_instance

# Define different prompt templates for different knowledge bases
PROMPT_TEMPLATES = {
    "ark_system_knowledge": """
    You are ARK, a helpful AI assistant with access to specific information about this Ubuntu system.
    Your task is to answer the user's question based *only* on the context provided below.
    If the context does not contain the answer, state that the information is not available in your knowledge base.
    Do not make up information. Be concise and accurate, citing the source file if possible.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """,
    "ark_project_knowledge": """
    You are ARK, a helpful AI assistant with access to a software project's source code and documentation.
    Your task is to answer the user's technical question based *only* on the context provided from the project files.
    Analyze the code snippets, documentation, and file contents to provide accurate answers.
    If the context does not contain the answer, state that the information is not available in the project's knowledge base.
    When referencing code, mention the source file from the context.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
}

def main(collection_name: str):
    """
    Main function to run the ARK assistant's query engine for a specific collection.
    """
    print(f"--- Initializing ARK Assistant for collection: '{collection_name}' ---")

    # --- 1. Define the RAG Prompt Template ---
    template = PROMPT_TEMPLATES.get(collection_name, PROMPT_TEMPLATES['ark_project_knowledge'])

    # --- 2. Construct the RAG Chain ---
    rag_chain = _rag_utils_instance.create_rag_chain(collection_name, template)

    print(f"\n--- ARK is ready. Querying '{collection_name}'. ---")
    print("Type 'exit' or 'quit' to end the session.")

    # --- 3. Interactive Query Loop with STREAMING ---
    while True:
        try:
            question = input("\n> You: ")
            if question.lower() in ['exit', 'quit']:
                print("\nARK shutting down. Goodbye.")
                break

            print("\n> ARK: ", end="", flush=True)

            full_response = ""
            for chunk in rag_chain.stream({"question": question}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()

        except KeyboardInterrupt:
            print("\n\nARK shutting down. Goodbye.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query ARK's knowledge base.")
    parser.add_argument("--collection", type=str, required=True, help="The name of the ChromaDB collection to query.")
    args = parser.parse_args()
    main(collection_name=args.collection)
