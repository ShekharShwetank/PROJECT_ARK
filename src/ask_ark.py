import argparse
import chromadb
import os
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM  # Corrected import
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
DB_PATH = "db"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
LLM_MODEL_NAME = "mistral"

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

    # --- 1. Initialize Vector Database and Retriever ---
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')

    print("Connecting to vector database...")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        print(f"Error: Collection '{collection_name}' not found. Please ingest data for it first using ingest.py.")
        return

    def retrieve_context(query_text, n_results=10):
        query_embedding = embedding_model.encode(query_text).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        context_with_metadata = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            source = meta.get('source', 'Unknown source')
            context_with_metadata.append(f"--- CONTEXT FROM: {source} ---\n{doc}")
        return "\n\n".join(context_with_metadata)

    # --- 2. Initialize the LLM ---
    print(f"Initializing LLM: {LLM_MODEL_NAME}...")
    llm = OllamaLLM(model=LLM_MODEL_NAME) # Using the corrected class name
    print("LLM Initialized.")

    # --- 3. Define the RAG Prompt Template ---
    template = PROMPT_TEMPLATES.get(collection_name, PROMPT_TEMPLATES['ark_project_knowledge'])
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # --- 4. Construct the RAG Chain ---
    rag_chain = (
        {"context": (lambda x: retrieve_context(x["question"])), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n--- ARK is ready. Querying '{collection_name}'. ---")
    print("Type 'exit' or 'quit' to end the session.")

    # --- 5. Interactive Query Loop with STREAMING ---
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
